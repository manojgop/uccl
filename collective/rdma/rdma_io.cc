#include "rdma_io.h"
#include "eqds.h"
#include "transport.h"
#include "transport_config.h"
#include "util/net.h"
#include "util/util.h"
#include "util_rdma.h"
#include "util_timer.h"
#include <glog/logging.h>
#include <infiniband/verbs.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>
#include <sys/mman.h>

namespace uccl {

std::shared_ptr<RDMAFactory> rdma_ctl;

static char uccl_ifname[MAX_IF_NAME_SIZE + 1];
static union socketAddress uccl_ifaddr;

int RDMAFactory::init_devs() {
  int num_devs;
  struct ibv_device** devices;
  std::vector<fs::path> gpu_cards;
  std::vector<std::pair<std::string, fs::path>> ib_nics;
  std::vector<std::tuple<std::string, fs::path, int>> ib_nics_with_dev_idx;
  std::stringstream init_devs_log;

#ifndef DISABLE_CALL_ONCE_STATIC
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
#endif
    if (rdma_ctl == NULL) {
      rdma_ctl = std::make_shared<RDMAFactory>();
    }
#ifndef DISABLE_CALL_ONCE_STATIC
  });
#endif
  //  Find interface for connection setup.
  int num_ifs = find_interfaces(uccl_ifname, &uccl_ifaddr, MAX_IF_NAME_SIZE, 1);
  if (num_ifs != 1) UCCL_INIT_CHECK(false, "No IP interface found");
  std::string oob_ip = get_dev_ip(uccl_ifname);
  LOG(INFO) << "Using OOB interface " << std::string(uccl_ifname) << " with IP "
            << oob_ip << " for connection setup";

  // Use UCCL_XXX first, if not set, use NCCL_XXX
  char* ib_hca = getenv("UCCL_IB_HCA");
  LOG(INFO) << "UCCL_IB_HCA: " << ib_hca;
  if (!ib_hca) {
    ib_hca = getenv("NCCL_IB_HCA");
    LOG(INFO) << "NCCL_IB_HCA: " << ib_hca;
  }

  struct ib_dev user_ib_ifs[MAX_IB_DEVS];
  bool searchNot = ib_hca && ib_hca[0] == '^';
  if (searchNot) ib_hca++;
  bool searchExact = ib_hca && ib_hca[0] == '=';
  if (searchExact) ib_hca++;

  int num_ib_ifs = parse_interfaces(ib_hca, user_ib_ifs, MAX_IB_DEVS);

  devices = ibv_get_device_list(&num_devs);
  if (devices == nullptr || num_devs == 0) {
    UCCL_LOG_ERROR << "Unable to get device list";
    goto error;
  }

  init_devs_log << "Found IB devices (ibv_get_device_list + NCCL_IB_HCA "
                   "filter, ordered by libibverbs):\n";
  if (rdma_ctl->num_devices == 0) {
    // When a single process creates multiple engines, the discovery can be
    // skipped
    for (int d = 0; d < num_devs && rdma_ctl->num_devices < MAX_IB_DEVS; d++) {
      struct ibv_context* context = ibv_open_device(devices[d]);
      if (context == nullptr) {
        UCCL_LOG_ERROR << "Unable to open device " << devices[d]->name;
        continue;
      }

      struct ibv_device_attr dev_attr;
      memset(&dev_attr, 0, sizeof(dev_attr));
      if (ibv_query_device(context, &dev_attr)) {
        ibv_close_device(context);
        continue;
      }

      for (int port_num = 1; port_num <= dev_attr.phys_port_cnt; port_num++) {
        struct ibv_port_attr port_attr;
        if (ibv_query_port(context, port_num, &port_attr)) {
          UCCL_LOG_ERROR << "Unable to query port_num " << port_num;
          ibv_close_device(context);
          continue;
        }

        // Check against user specified HCAs/ports
        if (!(match_if_list(devices[d]->name, port_num, user_ib_ifs, num_ib_ifs,
                            searchExact) ^
              searchNot)) {
          ibv_close_device(context);
          continue;
        }

        if (port_attr.state != IBV_PORT_ACTIVE) {
          ibv_close_device(context);
          continue;
        }

        // Initialize Dev
        struct FactoryDevice dev;
        strncpy(dev.ib_name, devices[d]->name, sizeof(devices[d]->name));

        dev.local_ip_str = oob_ip;
        dev.numa_node = get_dev_numa_node(dev.ib_name);
        dev.dev_attr = dev_attr;
        dev.port_attr = port_attr;
        dev.ib_port_num = port_num;

        double link_bw = (ncclIbSpeed(port_attr.active_speed) *
                          ncclIbWidth(port_attr.active_width)) *
                         1e6 / 8;
        dev.link_bw = link_bw;

        CHECK(ncclIbGetGidIndex(context, port_num, &port_attr, &dev.gid_idx));
        UCCL_LOG_RE << devices[d]->name << " uses gid_idx " << dev.gid_idx;

        if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
          dev.is_roce = true;
        } else if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
          dev.is_roce = false;
        } else {
          UCCL_LOG_ERROR << "Unknown link layer: " << port_attr.link_layer;
          ibv_close_device(context);
          continue;
        }

        dev.context = context;

        if (ibv_query_gid(context, port_num, dev.gid_idx, &dev.gid)) {
          UCCL_LOG_ERROR << "Unable to query GID";
          ibv_close_device(context);
          continue;
        }

        // Allocate a PD for this device
        dev.pd = ibv_alloc_pd(context);
        if (dev.pd == nullptr) {
          UCCL_LOG_ERROR << "Unable to allocate PD";
          ibv_close_device(context);
          continue;
        }

        // Detecting if the dev support extend cq.
        {
          struct ibv_cq_init_attr_ex cq_attr = {
              .cqe = 1,
              .comp_mask = 0,
          };
          struct ibv_cq_ex* cq_ex = ibv_create_cq_ex(context, &cq_attr);
          if (cq_ex) {
            UCCL_LOG_RE << "cq_ex supported on dev: " << devices[d]->name;
            ibv_destroy_cq(ibv_cq_ex_to_cq(cq_ex));
          } else {
            UCCL_LOG_RE << "cq_ex NOT supported on dev: " << devices[d]->name;
          }
          dev.support_cq_ex = cq_ex != nullptr;
        }

        // Detect UC support by trying to create a dummy UC QP
        {
          struct ibv_qp_init_attr qp_init_attr = {};
          qp_init_attr.qp_type = IBV_QPT_UC;
          qp_init_attr.send_cq = ibv_create_cq(context, 1, nullptr, nullptr, 0);
          qp_init_attr.recv_cq = qp_init_attr.send_cq;
          qp_init_attr.cap.max_send_wr = 1;
          qp_init_attr.cap.max_recv_wr = 1;
          qp_init_attr.cap.max_send_sge = 1;
          qp_init_attr.cap.max_recv_sge = 1;

          struct ibv_qp* uc_qp = ibv_create_qp(dev.pd, &qp_init_attr);
          if (uc_qp) {
            UCCL_LOG_RE << "UC supported on dev: " << devices[d]->name;
            ibv_destroy_qp(uc_qp);
            dev.support_uc = true;
          } else {
            UCCL_LOG_RE << "UC NOT supported on dev: " << devices[d]->name;
            dev.support_uc = false;
          }
          ibv_destroy_cq(qp_init_attr.send_cq);
        }

        // Detect DMA-BUF support
        {
          struct ibv_pd* pd = ibv_alloc_pd(context);
          if (pd == nullptr) {
            UCCL_LOG_ERROR << "Unable to allocate PD";
            ibv_close_device(context);
            continue;
          }

          // Test kernel DMA-BUF support with a dummy call (fd=-1)
          (void)ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/,
                                  0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
          dev.dma_buf_support =
              !((errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT));
          ibv_dealloc_pd(pd);

          UCCL_LOG_RE << "DMA-BUF support: " << dev.dma_buf_support
                      << " on dev: " << devices[d]->name;
        }

        init_devs_log << "\tdev_idx " << rdma_ctl->num_devices << ": "
                      << devices[d]->name << " (" << port_num << "/"
                      << (int)dev_attr.phys_port_cnt << ")\n";

        rdma_ctl->devices_.push_back(dev);
        UCCL_LOG_RE << "Initialized " << devices[d]->name
                    << " dev_idx: " << rdma_ctl->num_devices;

        rdma_ctl->num_devices++;
      }
    }
  }
  ibv_free_device_list(devices);

  // Get the GPUs, RDMA NICs, and their best mapping.
  {
    // Sorted by the GPU name.
    gpu_cards = get_gpu_cards();
    init_devs_log << "Found " << gpu_cards.size()
                  << " GPUs (get_gpu_cards, ordered by GPU rank):\n";
    int i = 0;
    for (auto const& gpu_card : gpu_cards) {
      init_devs_log << "\tGPU " << i++ << ": " << gpu_card.string() << "\n";
    }

    // Sorted by the RDMA NIC name.
    ib_nics = get_rdma_nics();
    for (auto const& [ib_name, ib_path] : ib_nics) {
      auto it =
          std::find_if(rdma_ctl->devices_.begin(), rdma_ctl->devices_.end(),
                       [ib_name](auto const& dev) {
                         return strcmp(dev.ib_name, ib_name.c_str()) == 0;
                       });
      // This ib_nic was excluded by the user.
      if (it == rdma_ctl->devices_.end()) continue;

      ib_nics_with_dev_idx.push_back(
          std::make_tuple(ib_name, ib_path, it - rdma_ctl->devices_.begin()));
    }
    // Sort by the dev_idx in rdma_ctl->devices_.
    std::sort(ib_nics_with_dev_idx.begin(), ib_nics_with_dev_idx.end(),
              [](auto const& a, auto const& b) {
                return std::get<2>(a) < std::get<2>(b);
              });
    init_devs_log << "Found " << ib_nics_with_dev_idx.size()
                  << " RDMA NICs (get_rdma_nics + NCCL_IB_HCA filter, ordered "
                     "by dev_idx in rdma_ctl->devices_[]):\n";
    for (auto const& [ib_name, ib_path, dev_idx] : ib_nics_with_dev_idx) {
      init_devs_log << "\tRDMA NIC " << ib_name << ": " << ib_path.string()
                    << ", dev_idx: " << dev_idx << "\n";
    }

    // Make sure both have the same number of NICs.
    CHECK(ib_nics_with_dev_idx.size() == rdma_ctl->num_devices);

    // Mapping GPU idx to dev_idx in rdma_ctl->devices_.
    rdma_ctl->gpu_to_dev_idx_ = map_gpu_to_dev(gpu_cards, ib_nics_with_dev_idx);
    init_devs_log << "Detected best GPU-NIC mapping: \n";
    for (auto const& [gpu_idx, dev_idx] : rdma_ctl->gpu_to_dev_idx_) {
      init_devs_log << "\tGPU " << gpu_idx << " -> NIC "
                    << std::get<0>(ib_nics_with_dev_idx[dev_idx])
                    << ", dev_idx: " << dev_idx << "\n";
    }
  }
  // Forcily output the log for better debugging in case of error.
  printf("%s", init_devs_log.str().c_str());

  return rdma_ctl->num_devices;

error:
  UCCL_INIT_CHECK(false, "Failed to initialize RDMAFactory");
}

/**
 * @brief Create a new RDMA context for a given device running on a specific
 * engine.
 *
 * @param dev
 * @param meta
 * @return RDMAContext*
 */
RDMAContext* RDMAFactory::CreateContext(TimerManager* rto,
                                        uint32_t* engine_unacked_bytes,
                                        eqds::EQDS* eqds, int dev,
                                        uint32_t engine_offset,
                                        union CtrlMeta meta,
                                        SharedIOContext* io_ctx) {
  RDMAContext* ctx = nullptr;

  if constexpr (kReceiverCCA == RECEIVER_CCA_EQDS)
    ctx = new EQDSRDMAContext(rto, engine_unacked_bytes, eqds, dev,
                              engine_offset, meta, io_ctx);
  else if constexpr (kSenderCCA == SENDER_CCA_TIMELY)
    ctx = new TimelyRDMAContext(rto, engine_unacked_bytes, eqds, dev,
                                engine_offset, meta, io_ctx);
  else if constexpr (kSenderCCA == SENDER_CCA_SWIFT)
    ctx = new SwiftRDMAContext(rto, engine_unacked_bytes, eqds, dev,
                               engine_offset, meta, io_ctx);

  CHECK(ctx != nullptr);
  return ctx;
}

std::pair<uint64_t, uint32_t> TXTracking::ack_rc_transmitted_chunks(
    void* subflow_context, RDMAContext* rdma_ctx, UINT_CSN csn, uint64_t now,
    uint32_t* flow_unacked_bytes, uint32_t* engine_outstanding_bytes) {
  auto* subflow = reinterpret_cast<SubUcclFlow*>(subflow_context);
  uint64_t tx_timestamp;
  uint32_t qpidx;

  uint32_t acked_bytes = 0;

  // Traverse unacked_chunks_
  // TODO: we can do more efficiently here.
  for (auto chunk = unacked_chunks_.begin(); chunk != unacked_chunks_.end();
       chunk++) {
    if (chunk->csn == csn.to_uint32()) {
      // We find it!
      chunk->ureq->send.acked_bytes += chunk->wr_ex->sge.length;

      acked_bytes += chunk->wr_ex->sge.length;

      if (chunk->ureq->send.acked_bytes == chunk->ureq->send.data_len) {
        auto poll_ctx = chunk->ureq->poll_ctx;
        // Wakeup app thread waiting one endpoint
        uccl_wakeup(poll_ctx);
        UCCL_LOG_IO << "RC TX message complete";
      }

      *flow_unacked_bytes -= chunk->wr_ex->sge.length;
      *engine_outstanding_bytes -= chunk->wr_ex->sge.length;

      tx_timestamp = chunk->timestamp;
      qpidx = chunk->wr_ex->qpidx;

      // Free wr_ex here.
      rdma_ctx->wr_ex_pool_->free_buff(
          reinterpret_cast<uint64_t>(chunk->wr_ex));

      unacked_chunks_.erase(chunk);
      break;
    }
  }

  auto newrtt_tsc = now - tx_timestamp;

  subflow->pcb.timely_cc.update_rate(now, newrtt_tsc, kEwmaAlpha);

  subflow->pcb.swift_cc.adjust_wnd(to_usec(newrtt_tsc, freq_ghz), acked_bytes);

  return std::make_pair(tx_timestamp, qpidx);
}

uint64_t TXTracking::ack_transmitted_chunks(void* subflow_context,
                                            RDMAContext* rdma_ctx,
                                            uint32_t num_acked_chunks,
                                            uint64_t t5, uint64_t t6,
                                            uint64_t remote_queueing_tsc,
                                            uint32_t* flow_unacked_bytes) {
  DCHECK(num_acked_chunks <= unacked_chunks_.size());

  auto* subflow = reinterpret_cast<SubUcclFlow*>(subflow_context);

  uint64_t t1 = 0;
  uint32_t seg_size = 0;

  while (num_acked_chunks) {
    auto& chunk = unacked_chunks_.front();
    if (chunk.last_chunk) {
      auto poll_ctx = chunk.ureq->poll_ctx;
      // Wakeup app thread waiting one endpoint
      uccl_wakeup(poll_ctx);
      UCCL_LOG_IO << "UC Tx message complete";
    }

    // Record timestamp of the oldest unacked chunk.
    if (t1 == 0) t1 = chunk.timestamp;

    seg_size += chunk.wr_ex->sge.length;

    *flow_unacked_bytes -= chunk.wr_ex->sge.length;

    // Free wr_ex here.
    rdma_ctx->wr_ex_pool_->free_buff(reinterpret_cast<uint64_t>(chunk.wr_ex));

    unacked_chunks_.erase(unacked_chunks_.begin());
    num_acked_chunks--;
  }

  if (unlikely(t5 <= t1)) {
    // Invalid timestamp.
    // We have found that t5 (transferred from NIC timestamp) may be
    // occasionally smaller than t1 (timestamp of the oldest unacked chunk).
    // When this happens, we use software timestamp to fix it.
    t5 = rdtsc();
  }

  auto endpoint_delay_tsc = t6 - t5 + remote_queueing_tsc;
  auto fabric_delay_tsc = (t6 - t1) - endpoint_delay_tsc;
  // Make RTT independent of segment size.
  auto serial_delay_tsc =
      us_to_cycles(seg_size * 1e6 / rdma_ctx->link_speed_, freq_ghz);
  if (fabric_delay_tsc > serial_delay_tsc ||
      to_usec(fabric_delay_tsc, freq_ghz) < kMAXRTTUS)
    fabric_delay_tsc -= serial_delay_tsc;
  else {
    // Invalid timestamp.
    // Recalculate delay.
    t5 = rdtsc();
    endpoint_delay_tsc = t6 - t5 + remote_queueing_tsc;
    fabric_delay_tsc = (t6 - t1) - endpoint_delay_tsc;
    if (fabric_delay_tsc > serial_delay_tsc)
      fabric_delay_tsc -= serial_delay_tsc;
    else {
      // This may be caused by clock synchronization.
      fabric_delay_tsc = 0;
    }
  }

  UCCL_LOG_IO << "Total: " << to_usec(t6 - t1, freq_ghz)
              << ", Endpoint delay: " << to_usec(endpoint_delay_tsc, freq_ghz)
              << ", Fabric delay: " << to_usec(fabric_delay_tsc, freq_ghz);

  // LOG_EVERY_N(INFO, 10000) << "Host: " <<
  // std::round(to_usec(endpoint_delay_tsc, freq_ghz)) <<
  //     ", Fabric: " << std::round(to_usec(fabric_delay_tsc, freq_ghz));

#ifdef TEST_TURNAROUND_ESTIMATION
  static bool first = true;
  static double avg_turnaround_delay = 0.0;
  static int count = 0;
  auto turnaround_delay = to_usec(remote_queueing_tsc, freq_ghz);

  if (turnaround_delay <
          500 /* filter wrong values (probabaly due to clock sync) */
      && count++ > 5000 /* warmup */) {
    if (first) {
      avg_turnaround_delay = turnaround_delay;
      first = false;
    } else {
      avg_turnaround_delay =
          (avg_turnaround_delay * count + turnaround_delay) / (count + 1);
    }
    LOG_EVERY_N(INFO, 1000)
        << "Turnaround delay: " << turnaround_delay
        << "us, Average turnaround delay: " << avg_turnaround_delay << "us";
  }
#endif

  if (fabric_delay_tsc) {
    // Update global cwnd.
    subflow->pcb.timely_cc.update_rate(t6, fabric_delay_tsc, kEwmaAlpha);
    // TODO: seperate enpoint delay and fabric delay.
    subflow->pcb.swift_cc.adjust_wnd(to_usec(fabric_delay_tsc, freq_ghz),
                                     seg_size);
  }

  return fabric_delay_tsc;
}

void SharedIOContext::check_ctrl_rq(bool force) {
  auto n_post_ctrl_rq = get_post_ctrl_rq_cnt();
  if (!force && n_post_ctrl_rq < kPostRQThreshold) return;

  int post_batch = std::min(kPostRQThreshold, (uint32_t)n_post_ctrl_rq);

  for (int i = 0; i < post_batch; i++) {
    auto chunk_addr = pop_ctrl_chunk();
    ctrl_recv_wrs_.recv_sges[i].addr = chunk_addr;

    CQEDesc* cqe_desc = pop_cqe_desc();
    cqe_desc->data = (uint64_t)chunk_addr;
    ctrl_recv_wrs_.recv_wrs[i].wr_id = (uint64_t)cqe_desc;
    ctrl_recv_wrs_.recv_wrs[i].next =
        (i == post_batch - 1) ? nullptr : &ctrl_recv_wrs_.recv_wrs[i + 1];
  }

  struct ibv_recv_wr* bad_wr;
  CHECK(ibv_post_recv(ctrl_qp_, &ctrl_recv_wrs_.recv_wrs[0], &bad_wr) == 0);
  UCCL_LOG_IO << "Posted " << post_batch << " recv requests for Ctrl QP";
  dec_post_ctrl_rq(post_batch);
}

void SharedIOContext::check_srq(bool force) {
  auto n_post_srq = get_post_srq_cnt();
  if (!force && n_post_srq < kPostRQThreshold) return;

  int post_batch = std::min(kPostRQThreshold, (uint32_t)n_post_srq);

  for (int i = 0; i < post_batch; i++) {
    if (!is_rc_mode()) {
      // UD/UC mode: Post buffers to SRQ
      uint64_t chunk_addr;
      uint32_t chunk_len;
      uint32_t lkey;

      if (is_ud_for_data()) {
        // UD uses GPU buffers for GPUDirect with MTU-limited size
        chunk_addr = pop_gpu_recv_chunk();
        chunk_len = gpu_recv_chunk_size_;  // Use actual UD chunk size, not kRetrChunkSize
        lkey = get_gpu_recv_chunk_lkey();
        
        // CRITICAL: Verify gpu_recv_chunk_size_ was initialized correctly
        CHECK(gpu_recv_chunk_size_ > 0 && gpu_recv_chunk_size_ <= 4096)
            << "BUG: gpu_recv_chunk_size_=" << gpu_recv_chunk_size_
            << " is invalid! Should be MTU-40 (e.g., 4056 for MTU 4096)";
        
        // CRITICAL: Verify chunk_addr is valid
        CHECK(chunk_addr != 0) << "BUG: pop_gpu_recv_chunk() returned NULL address!";
        
        if (i == 0) {
          VLOG(2) << "UD GPU recv buffer: addr=0x" << std::hex << chunk_addr
                  << std::dec << ", len=" << chunk_len << ", lkey=0x" << std::hex << lkey;
        }
      } else {
        // UC uses CPU buffers
        chunk_addr = pop_retr_chunk();
        chunk_len = kRetrChunkSize;
        lkey = get_retr_chunk_lkey();
      }

      if (is_ud_for_data()) {
        // UD requires 2 SGEs: GRH buffer (40 bytes) + data buffer
        // This matches how collective EFA works and fixes irdma driver issues
        // CRITICAL: Use paired GRH buffer based on GPU buffer index to avoid reuse
        uint64_t grh_buf = get_grh_buf_for_gpu_buf(chunk_addr);
        
        // SGE[0]: GRH header (40 bytes)
        dp_recv_wrs_.ud_sge_arrays[i][0].addr = grh_buf;
        dp_recv_wrs_.ud_sge_arrays[i][0].length = UD_ADDITION;  // 40 bytes
        dp_recv_wrs_.ud_sge_arrays[i][0].lkey = ud_grh_lkey_;
        
        // SGE[1]: Data payload (4056 bytes)
        dp_recv_wrs_.ud_sge_arrays[i][1].addr = chunk_addr;
        dp_recv_wrs_.ud_sge_arrays[i][1].length = chunk_len;
        dp_recv_wrs_.ud_sge_arrays[i][1].lkey = lkey;
        
        // CRITICAL: Validate buffer is within registered MR bounds
        uint64_t mr_base = (uint64_t)gpu_recv_mr_->addr;
        uint64_t mr_end = mr_base + gpu_recv_mr_->length;
        uint64_t buf_end = chunk_addr + chunk_len;
        
        CHECK(chunk_addr >= mr_base && buf_end <= mr_end)
            << "CRITICAL: Buffer 0x" << std::hex << chunk_addr 
            << " (len=" << std::dec << chunk_len << ") is outside MR range [0x"
            << std::hex << mr_base << ", 0x" << mr_end << ")"
            << ", offset=" << std::dec << (chunk_addr - mr_base)
            << ", lkey=0x" << std::hex << lkey << " vs MR lkey=0x" << gpu_recv_mr_->lkey;
        
        CHECK(lkey == gpu_recv_mr_->lkey)
            << "CRITICAL: lkey mismatch! Using 0x" << std::hex << lkey 
            << " but MR lkey is 0x" << gpu_recv_mr_->lkey;
        
        dp_recv_wrs_.recv_wrs[i].num_sge = 2;
        dp_recv_wrs_.recv_wrs[i].sg_list = dp_recv_wrs_.ud_sge_arrays[i];
        
        if (i == 0) {
          VLOG(2) << "UD SRQ WR[0]: SGE[0] addr=0x" << std::hex << grh_buf
                  << ", len=" << std::dec << UD_ADDITION << ", lkey=0x" << std::hex << ud_grh_lkey_
                  << "; SGE[1] addr=0x" << chunk_addr << ", len=" << std::dec << chunk_len
                  << ", lkey=0x" << std::hex << lkey;
        }
      } else {
        // UC uses single SGE
        dp_recv_wrs_.recv_sges[i].addr = chunk_addr;
        dp_recv_wrs_.recv_sges[i].length = chunk_len;
        dp_recv_wrs_.recv_sges[i].lkey = lkey;
        dp_recv_wrs_.recv_wrs[i].num_sge = 1;
        dp_recv_wrs_.recv_wrs[i].sg_list = &dp_recv_wrs_.recv_sges[i];
      }
      
      dp_recv_wrs_.recv_wrs[i].next =
          (i == post_batch - 1) ? nullptr : &dp_recv_wrs_.recv_wrs[i + 1];

      CQEDesc* cqe_desc = pop_cqe_desc();
      cqe_desc->data = (uint64_t)chunk_addr;
      dp_recv_wrs_.recv_wrs[i].wr_id = (uint64_t)cqe_desc;
    } else {
      // RC mode: No receive buffers needed
      dp_recv_wrs_.recv_wrs[i].num_sge = 0;
      dp_recv_wrs_.recv_wrs[i].sg_list = nullptr;
      dp_recv_wrs_.recv_wrs[i].next =
          (i == post_batch - 1) ? nullptr : &dp_recv_wrs_.recv_wrs[i + 1];
      dp_recv_wrs_.recv_wrs[i].wr_id = 0;
    }
  }

  struct ibv_recv_wr* bad_wr = nullptr;
  int ret = ibv_post_srq_recv(srq_, &dp_recv_wrs_.recv_wrs[0], &bad_wr);
  if (ret != 0) {
    LOG(ERROR) << "ibv_post_srq_recv failed with errno=" << ret << " (" << strerror(ret) << ")";
    LOG(ERROR) << "post_batch=" << post_batch << ", is_ud=" << is_ud_for_data()
               << ", gpu_recv_chunk_size_=" << gpu_recv_chunk_size_;
    
    if (bad_wr) {
      // Find which WR failed
      int failed_idx = -1;
      for (int i = 0; i < post_batch; i++) {
        if (&dp_recv_wrs_.recv_wrs[i] == bad_wr) {
          failed_idx = i;
          break;
        }
      }
      LOG(ERROR) << "Failed WR index: " << failed_idx;
      if (failed_idx >= 0) {
        LOG(ERROR) << "Failed WR num_sge=" << bad_wr->num_sge;
        if (is_ud_for_data() && bad_wr->num_sge == 2) {
          LOG(ERROR) << "  SGE[0]: addr=0x" << std::hex << dp_recv_wrs_.ud_sge_arrays[failed_idx][0].addr
                     << ", length=" << std::dec << dp_recv_wrs_.ud_sge_arrays[failed_idx][0].length
                     << ", lkey=0x" << std::hex << dp_recv_wrs_.ud_sge_arrays[failed_idx][0].lkey;
          LOG(ERROR) << "  SGE[1]: addr=0x" << std::hex << dp_recv_wrs_.ud_sge_arrays[failed_idx][1].addr
                     << ", length=" << std::dec << dp_recv_wrs_.ud_sge_arrays[failed_idx][1].length
                     << ", lkey=0x" << std::hex << dp_recv_wrs_.ud_sge_arrays[failed_idx][1].lkey;
        }
      }
    }
    CHECK(false) << "ibv_post_srq_recv failed!";
  }
  
  dec_post_srq(post_batch);
}

void SharedIOContext::flush_acks() {
  if (nr_tx_ack_wr_ == 0) return;

  tx_ack_wr_[nr_tx_ack_wr_ - 1].next = nullptr;

  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(ctrl_qp_, tx_ack_wr_, &bad_wr);
  DCHECK(ret == 0) << ret << ", nr_tx_ack_wr_: " << nr_tx_ack_wr_;

  UCCL_LOG_IO << "Flush " << nr_tx_ack_wr_ << " ACKs";

  inflight_ctrl_wrs_ += nr_tx_ack_wr_;

  nr_tx_ack_wr_ = 0;
}

int SharedIOContext::_poll_ctrl_cq_ex(void) {
  auto cq_ex = ctrl_cq_ex_;
  int work = 0;

  int budget = kMaxBatchCQ << 1;

  while (1) {
    struct ibv_poll_cq_attr poll_cq_attr = {};
    if (ibv_start_poll(cq_ex, &poll_cq_attr)) return work;
    int cq_budget = 0;

    while (1) {
      if (cq_ex->status != IBV_WC_SUCCESS) {
        CHECK(false) << "Ctrl CQ state error: " << cq_ex->status << ", "
                     << ibv_wc_read_opcode(cq_ex)
                     << ", ctrl_chunk_pool_size: " << ctrl_chunk_pool_->size();
      }

      CQEDesc* cqe_desc = reinterpret_cast<CQEDesc*>(cq_ex->wr_id);
      auto chunk_addr = (uint64_t)cqe_desc->data;

      auto opcode = ibv_wc_read_opcode(cq_ex);
      if (opcode == IBV_WC_RECV) {
        auto imm_data = ntohl(ibv_wc_read_imm_data(cq_ex));
        auto num_ack = imm_data;
        UCCL_LOG_IO << "Receive " << num_ack
                    << " ACKs, Chunk addr: " << chunk_addr
                    << ", byte_len: " << ibv_wc_read_byte_len(cq_ex);
        auto base_addr = chunk_addr + UD_ADDITION;
        for (int i = 0; i < num_ack; i++) {
          auto pkt_addr = base_addr + i * CtrlChunkBuffPool::kPktSize;

          auto* ucclsackh = reinterpret_cast<UcclSackHdr*>(pkt_addr);
          auto fid = ucclsackh->fid.value();
          auto peer_id = ucclsackh->peer_id.value();
          auto* rdma_ctx = find_rdma_ctx(peer_id, fid);

          rdma_ctx->uc_rx_ack<struct ibv_cq_ex>(cq_ex, ucclsackh);
        }
        inc_post_ctrl_rq();
      } else {
        inflight_ctrl_wrs_--;
      }

      push_ctrl_chunk(chunk_addr);

      push_cqe_desc(cqe_desc);

      if (++cq_budget == budget || ibv_next_poll(cq_ex)) break;

      if (opcode == IBV_WC_SEND) {
        // We don't count send WRs in budget.
        cq_budget--;
      }
    }
    ibv_end_poll(cq_ex);

    work += cq_budget;

    check_ctrl_rq(false);

    if (cq_budget < budget) break;
  }

  return work;
}

int SharedIOContext::_poll_ctrl_cq_normal(void) {
  auto cq = ibv_cq_ex_to_cq(ctrl_cq_ex_);
  struct ibv_wc wcs[kMaxBatchCQ];

  int cq_budget = 0;
  int budget = kMaxBatchCQ << 1;

  while (1) {
    int nr_wcs = ibv_poll_cq(cq, kMaxBatchCQ, wcs);
    if (nr_wcs == 0) break;

    for (int i = 0; i < nr_wcs; i++) {
      auto* wc = wcs + i;
      DCHECK(wc->status == IBV_WC_SUCCESS)
          << "Ctrl CQ state error: " << wc->status;
      CQEDesc* cqe_desc = (CQEDesc*)wc->wr_id;
      auto chunk_addr = (uint64_t)cqe_desc->data;

      auto opcode = wc->opcode;

      if (opcode == IBV_WC_RECV) {
        auto imm_data = ntohl(wc->imm_data);
        auto num_ack = imm_data;
        UCCL_LOG_IO << "Receive " << num_ack
                    << " ACKs, Chunk addr: " << chunk_addr
                    << ", byte_len: " << wc->byte_len;
        auto base_addr = chunk_addr + UD_ADDITION;
        for (int i = 0; i < num_ack; i++) {
          auto pkt_addr = base_addr + i * CtrlChunkBuffPool::kPktSize;

          auto* ucclsackh = reinterpret_cast<UcclSackHdr*>(pkt_addr);
          auto fid = ucclsackh->fid.value();
          auto peer_id = ucclsackh->peer_id.value();
          auto* rdma_ctx = find_rdma_ctx(peer_id, fid);

          rdma_ctx->uc_rx_ack<struct ibv_wc>(wc, ucclsackh);
        }
        inc_post_ctrl_rq();
      } else {
        inflight_ctrl_wrs_--;
      }

      push_ctrl_chunk(chunk_addr);

      push_cqe_desc(cqe_desc);

      if (opcode == IBV_WC_SEND) {
        // We don't count send WRs in budget.
        cq_budget--;
      }
    }

    cq_budget += nr_wcs;

    check_ctrl_rq(false);

    if (cq_budget >= budget) break;
  }

  return cq_budget;
}

int SharedIOContext::_rc_poll_recv_cq_ex(void) {
  auto cq_ex = recv_cq_ex_;
  int cq_budget = 0;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  while (1) {
    if (cq_ex->status != IBV_WC_SUCCESS) {
      CHECK(false) << "data path CQ state error: " << cq_ex->status
                   << " from QP:" << ibv_wc_read_qp_num(cq_ex);
    }

    auto* rdma_ctx = qpn_to_rdma_ctx(ibv_wc_read_qp_num(cq_ex));

    rdma_ctx->rc_rx_chunk<struct ibv_cq_ex>(cq_ex);

    inc_post_srq();

    if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
  }

  ibv_end_poll(cq_ex);

  return cq_budget;
}

int SharedIOContext::_rc_poll_send_cq_ex(void) {
  auto cq_ex = send_cq_ex_;
  int cq_budget = 0;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  while (1) {
    if (cq_ex->status != IBV_WC_SUCCESS) {
      CHECK(false) << "data path CQ state error: " << cq_ex->status
                   << " from QP:" << ibv_wc_read_qp_num(cq_ex);
    }

    auto* rdma_ctx = qpn_to_rdma_ctx(ibv_wc_read_qp_num(cq_ex));

    rdma_ctx->rc_rx_ack<struct ibv_cq_ex>(cq_ex);

    if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
  }
  ibv_end_poll(cq_ex);

  return cq_budget;
}

int SharedIOContext::_uc_poll_send_cq_ex(void) {
  auto cq_ex = send_cq_ex_;
  int cq_budget = 0;
  int budget = kMaxBatchCQ << 1;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  while (1) {
    if (cq_ex->status != IBV_WC_SUCCESS) {
      CHECK(false) << "data path CQ state error: " << cq_ex->status
                   << " from QP:" << ibv_wc_read_qp_num(cq_ex);
    }

    auto* cqe_desc = (CQEDesc*)cq_ex->wr_id;

    if (cqe_desc) {
      // Completion signal from rtx.
      auto retr_hdr = (uint64_t)cqe_desc->data;
      push_retr_hdr(retr_hdr);
      push_cqe_desc(cqe_desc);
    }

    if (++cq_budget == budget || ibv_next_poll(cq_ex)) break;
  }

  ibv_end_poll(cq_ex);

  return cq_budget;
}

int SharedIOContext::_uc_poll_recv_cq_ex(void) {
  auto cq_ex = recv_cq_ex_;
  int cq_budget = 0;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  std::vector<RDMAContext*> rdma_ctxs;

  while (1) {
    if (cq_ex->status != IBV_WC_SUCCESS) {
      CHECK(false) << "data path CQ state error: " << cq_ex->status
                   << " from QP:" << ibv_wc_read_qp_num(cq_ex);
    }

    auto* rdma_ctx = qpn_to_rdma_ctx(ibv_wc_read_qp_num(cq_ex));

    auto* cqe_desc = (CQEDesc*)cq_ex->wr_id;
    auto chunk_addr = (uint64_t)cqe_desc->data;
    auto opcode = ibv_wc_read_opcode(cq_ex);

    if (likely(opcode == IBV_WC_RECV_RDMA_WITH_IMM)) {
      // Common case.
      rdma_ctx->uc_rx_chunk<struct ibv_cq_ex>(cq_ex);
    } else {
      // Rare case.
      rdma_ctx->uc_rx_rtx_chunk<struct ibv_cq_ex>(cq_ex, chunk_addr);
    }

    rdma_ctxs.push_back(rdma_ctx);

    push_retr_chunk(chunk_addr);

    push_cqe_desc(cqe_desc);

    inc_post_srq();

    if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
  }
  ibv_end_poll(cq_ex);

  for (auto rdma_ctx : rdma_ctxs) {
    rdma_ctx->uc_post_acks();
  }

  flush_acks();

  return cq_budget;
}

int SharedIOContext::_rc_poll_send_cq_normal(void) {
  struct ibv_wc wcs[kMaxBatchCQ];

  auto* cq = ibv_cq_ex_to_cq(send_cq_ex_);

  int nr_wcs = ibv_poll_cq(cq, kMaxBatchCQ, wcs);

  for (int i = 0; i < nr_wcs; i++) {
    auto* wc = wcs + i;
    DCHECK(wc->status == IBV_WC_SUCCESS)
        << "RC send CQ state error: " << wc->status << ", " << wc->byte_len;
    auto* rdma_ctx = qpn_to_rdma_ctx(wc->qp_num);
    rdma_ctx->rc_rx_ack<struct ibv_wc>(wc);
  }

  return nr_wcs;
}

int SharedIOContext::_rc_poll_recv_cq_normal(void) {
  struct ibv_wc wcs[kMaxBatchCQ];

  auto* cq = ibv_cq_ex_to_cq(recv_cq_ex_);

  int nr_wcs = ibv_poll_cq(cq, kMaxBatchCQ, wcs);

  for (int i = 0; i < nr_wcs; i++) {
    auto* wc = wcs + i;
    DCHECK(wc->status == IBV_WC_SUCCESS)
        << "RC recv CQ state error: " << wc->status;
    auto* cqe_desc = (CQEDesc*)wc->wr_id;
    auto* rdma_ctx = qpn_to_rdma_ctx(wc->qp_num);
    rdma_ctx->rc_rx_chunk<struct ibv_wc>(wc);

    inc_post_srq();
  }

  return nr_wcs;
}

int SharedIOContext::_uc_poll_send_cq_normal(void) {
  struct ibv_wc wcs[kMaxBatchCQ];

  auto* cq = ibv_cq_ex_to_cq(send_cq_ex_);

  int nr_wcs = ibv_poll_cq(cq, kMaxBatchCQ, wcs);

  for (int i = 0; i < nr_wcs; i++) {
    auto* wc = wcs + i;
    DCHECK(wc->status == IBV_WC_SUCCESS)
        << "UC send CQ state error: " << wc->status;
    auto* cqe_desc = (CQEDesc*)wc->wr_id;

    if (cqe_desc) {
      // Completion signal from rtx.
      auto retr_hdr = (uint64_t)cqe_desc->data;
      push_retr_hdr(retr_hdr);
      push_cqe_desc(cqe_desc);
    }
  }

  return nr_wcs;
}

int SharedIOContext::_uc_poll_recv_cq_normal(void) {
  struct ibv_wc wcs[kMaxBatchCQ];

  auto* cq = ibv_cq_ex_to_cq(recv_cq_ex_);

  int nr_wcs = ibv_poll_cq(cq, kMaxBatchCQ, wcs);

  std::vector<RDMAContext*> rdma_ctxs;

  for (int i = 0; i < nr_wcs; i++) {
    auto* wc = wcs + i;
    DCHECK(wc->status == IBV_WC_SUCCESS)
        << "UC recv CQ state error: " << wc->status;
    auto* cqe_desc = (CQEDesc*)wc->wr_id;
    auto* rdma_ctx = qpn_to_rdma_ctx(wc->qp_num);

    auto chunk_addr = (uint64_t)cqe_desc->data;
    auto opcode = wc->opcode;

    if (likely(opcode == IBV_WC_RECV_RDMA_WITH_IMM)) {
      // Common case.
      rdma_ctx->uc_rx_chunk<struct ibv_wc>(wc);
    } else {
      // Rare case.
      rdma_ctx->uc_rx_rtx_chunk<struct ibv_wc>(wc, chunk_addr);
    }

    rdma_ctxs.push_back(rdma_ctx);

    push_retr_chunk(chunk_addr);

    push_cqe_desc(cqe_desc);

    inc_post_srq();
  }

  for (auto rdma_ctx : rdma_ctxs) {
    rdma_ctx->uc_post_acks();
  }

  flush_acks();

  return nr_wcs;
}

// UD-specific poll functions for data path
int SharedIOContext::_ud_poll_send_cq_ex(void) {
  // WORKAROUND: irdma driver bug with UD+SRQ+CQ_EX - poll send_cq for BOTH send and recv
  // Discriminate by opcode: IBV_WC_SEND=0 vs IBV_WC_RECV=128
  auto cq_ex = send_cq_ex_;
  int cq_budget = 0;
  int budget = kMaxBatchCQ << 1;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  std::vector<RDMAContext*> rdma_ctxs;
  std::vector<CQEDesc*> recv_cqe_descs;  // For deferred return after sync
  std::vector<uint64_t> recv_chunk_addrs;  // For deferred return after sync
  static bool first_cqe = true;

  while (1) {
    if (cq_ex->status != IBV_WC_SUCCESS) {
      auto qp_num = ibv_wc_read_qp_num(cq_ex);
      auto opcode = ibv_wc_read_opcode(cq_ex);
      auto vendor_err = ibv_wc_read_vendor_err(cq_ex);
      LOG(ERROR) << "UD CQ error: status=" << cq_ex->status
                 << ", QP=" << qp_num << ", opcode=" << opcode
                 << ", vendor_err=" << vendor_err;
      CHECK(false) << "UD CQ state error: " << cq_ex->status;
    }

    auto opcode = ibv_wc_read_opcode(cq_ex);
    auto* cqe_desc = (CQEDesc*)cq_ex->wr_id;

    if (first_cqe) {
      auto qp_num = ibv_wc_read_qp_num(cq_ex);
      LOG(INFO) << "UD FIRST CQE: opcode=" << opcode << " (IBV_WC_SEND=0, IBV_WC_RECV=128), QP=" << qp_num << ", CQ=" << cq_ex;
      first_cqe = false;
    }

    if (opcode == IBV_WC_SEND) {
      // Send completion
      if (cqe_desc) {
        // Distinguish between normal send and retransmission using type field
        if (cqe_desc->type == 1) {
          // Retransmission completion - free retr_hdr
          auto retr_hdr = (uint64_t)cqe_desc->data;
          push_retr_hdr(retr_hdr);
        }
        // Note: Normal send completions (type==0) contain wr_ex pointer
        // The wr_ex is freed when timing wheel entry is removed or on ack
        push_cqe_desc(cqe_desc);
      }
    } else if (opcode == IBV_WC_RECV) {
      // Receive completion (on same CQ due to driver bug)
      auto qp_num = ibv_wc_read_qp_num(cq_ex);
      auto* rdma_ctx = qpn_to_rdma_ctx(qp_num);
      auto chunk_addr = (uint64_t)cqe_desc->data;

      static int recv_count = 0;
      if (++recv_count <= 5) {
        LOG(INFO) << "UD RECV #" << recv_count << ": QP=" << qp_num << ", chunk_addr=" << std::hex << chunk_addr << std::dec;
      }

      // UD: Both normal and retransmission arrive as IBV_WC_RECV in GPU buffer
      rdma_ctx->ud_rx_chunk<struct ibv_cq_ex>(cq_ex);

      rdma_ctxs.push_back(rdma_ctx);
      recv_cqe_descs.push_back(cqe_desc);  // Save for later
      recv_chunk_addrs.push_back(chunk_addr);  // Save for later
    } else {
      LOG(FATAL) << "UD CQ unexpected opcode: " << opcode;
    }

    if (++cq_budget == budget || ibv_next_poll(cq_ex)) break;
  }

  ibv_end_poll(cq_ex);

  // CRITICAL: Sync BEFORE returning buffers
  if (!rdma_ctxs.empty()) {
#ifndef __HIP_PLATFORM_AMD__
    cudaStreamSynchronize(0);
#else
    hipStreamSynchronize(0);
#endif

    // NOW safe to return buffers after sync
    for (size_t i = 0; i < recv_chunk_addrs.size(); i++) {
      push_gpu_recv_chunk(recv_chunk_addrs[i]);
      push_cqe_desc(recv_cqe_descs[i]);
      inc_post_srq();
    }
  }

  // Post recv WRs back to SRQ if we handled any receives
  if (!rdma_ctxs.empty()) {
    check_srq(false);

    // Post ACKs for all flows that received data
    for (auto rdma_ctx : rdma_ctxs) {
      rdma_ctx->ud_post_acks();
    }

    flush_acks();
  }

  return cq_budget;
}

int SharedIOContext::_ud_poll_recv_cq_ex(void) {
  // WORKAROUND: irdma driver bug - don't poll recv_cq, all completions on send_cq
  // This function is now a no-op, all work done in _ud_poll_send_cq_ex
  return 0;
}

int SharedIOContext::_ud_poll_send_cq_normal(void) {
  struct ibv_wc wcs[kMaxBatchCQ];
  auto* cq = ibv_cq_ex_to_cq(send_cq_ex_);
  int nr_wcs = ibv_poll_cq(cq, kMaxBatchCQ, wcs);

  for (int i = 0; i < nr_wcs; i++) {
    auto* wc = wcs + i;
    DCHECK(wc->status == IBV_WC_SUCCESS)
        << "UD send CQ state error: " << wc->status;
    auto* cqe_desc = (CQEDesc*)wc->wr_id;
    auto opcode = wc->opcode;

    // Send CQ should only see send completions
    DCHECK(opcode == IBV_WC_SEND) << "UD send_cq unexpected opcode: " << opcode
                                   << " (expected IBV_WC_SEND=0)";

    if (cqe_desc) {
      if (cqe_desc->type == 1) {
        // Retransmission completion
        auto retr_hdr = (uint64_t)cqe_desc->data;
        push_retr_hdr(retr_hdr);
      }
      push_cqe_desc(cqe_desc);
    }
  }

  return nr_wcs;
}

int SharedIOContext::_ud_poll_recv_cq_normal(void) {
  struct ibv_wc wcs[kMaxBatchCQ];
  auto* cq = ibv_cq_ex_to_cq(recv_cq_ex_);
  int nr_wcs = ibv_poll_cq(cq, kMaxBatchCQ, wcs);

  std::vector<RDMAContext*> rdma_ctxs;
  int successful_wcs = 0;

  for (int i = 0; i < nr_wcs; i++) {
    auto* wc = wcs + i;
    if (wc->status != IBV_WC_SUCCESS) {
      auto* cqe_desc = (CQEDesc*)wc->wr_id;
      auto chunk_addr = (uint64_t)cqe_desc->data;
      
      LOG(ERROR) << "UD recv CQ error - status: " << wc->status
                 << " (" << ibv_wc_status_str(wc->status) << ")"
                 << ", vendor_err: " << wc->vendor_err << " (0x" << std::hex << wc->vendor_err << ")"
                 << ", qp_num: " << std::dec << wc->qp_num
                 << ", opcode: " << wc->opcode
                 << ", byte_len: " << wc->byte_len
                 << ", wr_id: 0x" << std::hex << wc->wr_id
                 << ", chunk_addr: 0x" << chunk_addr
                 << ", recv_cq=" << std::dec << (void*)recv_cq_ex_
                 << ", srq_posted=" << (kMaxSRQ - get_post_srq_cnt());
      
      // Provide detailed diagnostics for buffer management bugs
      uint64_t chunk_offset = chunk_addr - (uint64_t)gpu_recv_buf_;
      uint32_t buffer_index = chunk_offset / gpu_recv_chunk_size_;
      uint64_t page_base = chunk_addr & ~0xFFFULL;  // 4KB page alignment
      uint64_t offset_in_page = chunk_addr & 0xFFFULL;
      
      LOG(ERROR) << "Buffer state: gpu_recv_chunk_size_=" << gpu_recv_chunk_size_
                 << ", gpu_recv_buf_base=0x" << std::hex << (uint64_t)gpu_recv_buf_
                 << ", chunk_offset=" << std::dec << chunk_offset
                 << ", buffer_index=" << buffer_index
                 << ", page_base=0x" << std::hex << page_base
                 << ", offset_in_page=" << std::dec << offset_in_page
                 << ", alignment=" << (chunk_addr % 64) << "B";
      
      LOG(ERROR) << "MR info: addr=0x" << std::hex << (uint64_t)gpu_recv_mr_->addr
                 << ", length=" << std::dec << gpu_recv_mr_->length
                 << ", lkey=0x" << std::hex << gpu_recv_mr_->lkey
                 << ", rkey=0x" << gpu_recv_mr_->rkey;
      
      push_gpu_recv_chunk(chunk_addr);
      push_cqe_desc(cqe_desc);
      inc_post_srq();
      
      // vendor_err 131073 (0x20001) is an Intel irdma driver error
      // Likely causes:
      // 1. GPU managed memory (cudaMallocManaged) not fully supported by irdma for DMA
      // 2. Driver bug with specific buffer indices or alignments
      // 3. Memory access violation at driver/hardware level
      LOG(FATAL) << "Intel irdma vendor_err 131073 at buffer_index=" << buffer_index
                 << " - This appears to be an irdma driver bug with GPU managed memory."
                 << " Consider using regular GPU memory (cudaMalloc) instead of cudaMallocManaged,"
                 << " or switching to per-QP RQ instead of SRQ (like collective/efa)";
    }
    
    // Process successful completion
    auto* cqe_desc = (CQEDesc*)wc->wr_id;
    auto* rdma_ctx = qpn_to_rdma_ctx(wc->qp_num);
    auto chunk_addr = (uint64_t)cqe_desc->data;
    auto opcode = wc->opcode;

    // Recv CQ should only see receive completions
    DCHECK(opcode == IBV_WC_RECV) << "UD recv_cq unexpected opcode: " << opcode
                                   << " (expected IBV_WC_RECV=128)";

    // UD: Both normal and retransmission arrive as IBV_WC_RECV in GPU buffer
    rdma_ctx->ud_rx_chunk<struct ibv_wc>(wc);

    rdma_ctxs.push_back(rdma_ctx);
    successful_wcs++;

    // Store buffer info for later return (after sync)
    // We'll return buffers after GPU sync completes
  }

  // CRITICAL: Sync BEFORE returning buffers to ensure GPU copies complete
  // If we return buffers first, check_srq() immediately reposts them and they get
  // overwritten while the GPU copy is still in flight!
  if (successful_wcs > 0) {
#ifndef __HIP_PLATFORM_AMD__
    cudaStreamSynchronize(0);
#else
    hipStreamSynchronize(0);
#endif

    // NOW safe to return buffers after sync
    for (int i = 0; i < nr_wcs; i++) {
      auto* wc = wcs + i;
      if (wc->status != IBV_WC_SUCCESS) continue;  // Already handled in error path above
      auto* cqe_desc = (CQEDesc*)wc->wr_id;
      auto chunk_addr = (uint64_t)cqe_desc->data;
      
      push_gpu_recv_chunk(chunk_addr);
      push_cqe_desc(cqe_desc);
      inc_post_srq();  // CRITICAL: Must increment to trigger check_srq()
    }
  }

  // Post recv WRs back to SRQ
  check_srq(false);

  // Post ACKs for all flows that received data
  for (auto rdma_ctx : rdma_ctxs) {
    rdma_ctx->ud_post_acks();
  }

  flush_acks();

  return nr_wcs;
}

void serialize_fifo_item(FifoItem const& item, char* buf) {
  static_assert(sizeof(FifoItem) == 64, "FifoItem must be 64 bytes");

  std::memcpy(buf + 0, &item.addr, sizeof(uint64_t));
  std::memcpy(buf + 8, &item.size, sizeof(uint32_t));
  std::memcpy(buf + 12, &item.rkey, sizeof(uint32_t));
  std::memcpy(buf + 16, &item.nmsgs, sizeof(uint32_t));
  std::memcpy(buf + 20, &item.rid, sizeof(uint32_t));
  std::memcpy(buf + 24, &item.idx, sizeof(uint64_t));
  std::memcpy(buf + 32, &item.engine_offset, sizeof(uint32_t));
  std::memcpy(buf + 36, &item.padding, sizeof(item.padding));
}

void deserialize_fifo_item(char const* buf, FifoItem* item) {
  std::memcpy(&item->addr, buf + 0, sizeof(uint64_t));
  std::memcpy(&item->size, buf + 8, sizeof(uint32_t));
  std::memcpy(&item->rkey, buf + 12, sizeof(uint32_t));
  std::memcpy(&item->nmsgs, buf + 16, sizeof(uint32_t));
  std::memcpy(&item->rid, buf + 20, sizeof(uint32_t));
  std::memcpy(&item->idx, buf + 24, sizeof(uint64_t));
  std::memcpy(&item->engine_offset, buf + 32, sizeof(uint32_t));
  std::memcpy(&item->padding, buf + 36, sizeof(item->padding));
}

}  // namespace uccl
