#include "transport.h"
#include "transport_config.h"
#include "util/util.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <deque>
#include <thread>
#include <signal.h>
#include <sstream>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

using namespace uccl;

// Helper functions for persistent OOB coordination
inline int create_oob_client_socket(const std::string& server_ip, int port) {
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  CHECK(sockfd >= 0) << "Failed to create OOB socket";
  
  struct sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port + 1);  // Use port+1 for persistent connection
  server_addr.sin_addr.s_addr = inet_addr(server_ip.c_str());
  
  while (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  
  int flag = 1;
  setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (void*)&flag, sizeof(int));
  return sockfd;
}

inline int create_oob_server_socket(int port) {
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  CHECK(sockfd >= 0) << "Failed to create OOB listen socket";
  
  int opt = 1;
  setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  
  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port + 1);  // Use port+1 for persistent connection
  addr.sin_addr.s_addr = INADDR_ANY;
  
  CHECK(bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) >= 0) << "Bind failed";
  CHECK(listen(sockfd, 1) >= 0) << "Listen failed";
  
  struct sockaddr_in client_addr;
  socklen_t client_len = sizeof(client_addr);
  int client_fd = accept(sockfd, (struct sockaddr*)&client_addr, &client_len);
  CHECK(client_fd >= 0) << "Accept failed";
  
  int flag = 1;
  setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, (void*)&flag, sizeof(int));
  
  close(sockfd);
  return client_fd;
}

const std::chrono::duration kReportIntervalSec = std::chrono::seconds(2);
const size_t kReportIters = 5000;
const uint32_t kNumConns = 4;
const size_t kChunkSize = 131072;  // 128KB chunk size - matches NCCL_P2P_NET_CHUNKSIZE

size_t kTestMsgSize = 131072;  // 128KB - matches NCCL_P2P_NET_CHUNKSIZE
size_t kMaxInflight = 8;

DEFINE_uint64(size, 131072, "Total transfer size per iteration. Will be chunked into 128KB pieces.");
DEFINE_uint64(iterations, 7500, "Number of iterations (safe for 128KB without delay).");
DEFINE_uint64(delay_us, 0, "Delay in us after each send. Use 300us for 100K+ iterations.");
DEFINE_uint64(infly, 64, "Max inflight messages. Use 64+ for peak tput, 8 for latency.");
DEFINE_string(serverip, "", "Server IP address the client tries to connect.");
DEFINE_string(clientip, "", "Client IP address the server tries to connect.");
DEFINE_uint32(oobport, 19999, "Out-of-band TCP port for bootstrapping and port exchange.");
DEFINE_bool(verify, false, "Whether to check data correctness.");
DEFINE_bool(rand, false, "Whether to use randomized data length.");
DEFINE_string(
    test, "basic",
    "Which test to run: basic, async, pingpong, mt (multi-thread), "
    "mc (multi-connection), mq (multi-queue), bimq (bi-directional mq), tput.");

enum TestType { kBasic, kAsync, kPingpong, kMt, kMc, kMq, kBiMq, kTput };

uint64_t* get_host_ptr(uint64_t* dev_ptr, size_t size) {
  uint64_t* host_ptr = (uint64_t*)malloc(size);
  cudaMemcpy(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost);
  return host_ptr;
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  kTestMsgSize = FLAGS_size;
  kMaxInflight = FLAGS_infly;

  bool is_client;
  if (!FLAGS_serverip.empty()) {
    is_client = true;
  } else if (!FLAGS_clientip.empty()) {
    is_client = false;
  } else {
    LOG(FATAL)
        << "Please specify server IP or client IP, and only one of them.";
  }

  TestType test_type;
  if (FLAGS_test == "basic") {
    test_type = kBasic;
  } else if (FLAGS_test == "async") {
    test_type = kAsync;
  } else if (FLAGS_test == "pingpong") {
    test_type = kPingpong;
  } else if (FLAGS_test == "mt") {
    test_type = kMt;
  } else if (FLAGS_test == "mc") {
    test_type = kMc;
  } else if (FLAGS_test == "mq") {
    test_type = kMq;
  } else if (FLAGS_test == "bimq") {
    test_type = kBiMq;
  } else if (FLAGS_test == "tput") {
    test_type = kTput;
  } else {
    LOG(FATAL) << "Unknown test type: " << FLAGS_test;
  }

  std::mt19937 generator(42);
  std::uniform_int_distribution<int> distribution(1024, kTestMsgSize);
  srand(42);
  pin_thread_to_cpu(0);
  
  // Use FLAGS_infly for max inflight depth
  kMaxInflight = FLAGS_infly;

  // OOB socket for coordination (used in kTput test)
  int oob_sock = -1;

  if (is_client) {
    auto ep = Endpoint();
    ep.initialize_engine_by_gpu_idx(0);
    DCHECK(FLAGS_serverip != "");
    int const kMaxArraySize = std::max(kNumConns, kNumVdevices);
    ConnID conn_id, conn_id2;
    ConnID conn_id_vec[kMaxArraySize];
    int remote_vdevs[kMaxArraySize];
    std::string remote_ip[kMaxArraySize];

    for (int i = 0; i < kMaxArraySize; i++) ep.uccl_listen();

    // Exchange listen ports with server via out-of-band TCP connection
    std::vector<uint16_t> remote_listen_ports(kMaxArraySize);
    LOG(INFO) << "[Client] Exchanging listen ports via OOB connection on port " << FLAGS_oobport;
    connect_exchange(FLAGS_oobport, FLAGS_serverip, ep.listen_port_vec_.data(),
                     ep.listen_port_vec_.size() * sizeof(uint16_t),
                     remote_listen_ports.data(),
                     kMaxArraySize * sizeof(uint16_t));
    LOG(INFO) << "[Client] Received server ports, preparing CUDA buffers...";

    // Allocate and register CUDA memory BEFORE connecting.
    // This prevents the client from flooding the server with packets
    // before the server has called uccl_recv().
    int send_len = kTestMsgSize, recv_len = kTestMsgSize;
    // Use synchronous mode (pipeline depth 1) to avoid buffer reuse issues
    // With async APIs, buffers may still be in use even after poll() returns
    constexpr int kPipelineDepth = 1;
    constexpr int kNumBuffers = 8;  // Allocate extra buffers for safety
    uint8_t *data[kNumBuffers], *data2[kNumBuffers];
    Mhandle mh[kNumBuffers], mh2[kNumBuffers];

    auto gpu_idx = 0;  // Use single GPU for all buffers
    auto dev_idx = get_dev_idx_by_gpu_idx(0);
    cudaSetDevice(gpu_idx);
    auto* dev = EFAFactory::GetEFADevice(dev_idx);

    for (int i = 0; i < kNumBuffers; i++) {

#ifdef INTEL_RDMA_NIC
      cudaMallocManaged(&data[i], kTestMsgSize, cudaMemAttachGlobal);
      mh[i].mr =
          ibv_reg_mr(dev->pd, data[i], kTestMsgSize,
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
      cudaMallocManaged(&data2[i], kTestMsgSize, cudaMemAttachGlobal);
      mh2[i].mr =
          ibv_reg_mr(dev->pd, data2[i], kTestMsgSize,
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
#else
      cudaMalloc(&data[i], kTestMsgSize);
      mh[i].mr =
          ibv_reg_mr(dev->pd, data[i], kTestMsgSize, IBV_ACCESS_LOCAL_WRITE);
      cudaMalloc(&data2[i], kTestMsgSize);
      mh2[i].mr =
          ibv_reg_mr(dev->pd, data2[i], kTestMsgSize, IBV_ACCESS_LOCAL_WRITE);
#endif
    }
    cudaSetDevice(0);

    LOG(INFO) << "[Client] CUDA buffers ready, connecting to server...";

    // Create persistent OOB socket for coordination (only for kTput test)
    if (test_type == kTput) {
      LOG(INFO) << "[Client] Creating persistent OOB connection for flow control...";
      oob_sock = create_oob_client_socket(FLAGS_serverip, FLAGS_oobport);
      LOG(INFO) << "[Client] OOB connection established";
    }

    conn_id = ep.uccl_connect(0, 0, FLAGS_serverip, remote_listen_ports[0]);
    
    if (test_type == kMc) {
      conn_id_vec[0] = conn_id;
      for (int i = 1; i < kNumConns; i++)
        conn_id_vec[i] =
            ep.uccl_connect(0, 0, FLAGS_serverip, remote_listen_ports[0]);
    } else if (test_type == kMq) {
      conn_id_vec[0] = conn_id;
      for (int i = 1; i < kNumVdevices; i++)
        conn_id_vec[i] =
            ep.uccl_connect(i, i, FLAGS_serverip, remote_listen_ports[i]);
    } else if (test_type == kBiMq) {
      conn_id_vec[0] = conn_id;
      for (int i = 1; i < kNumVdevices; i++) {
        if (i % 2 == 0)
          conn_id_vec[i] =
              ep.uccl_connect(i, i, FLAGS_serverip, remote_listen_ports[i]);
        else
          conn_id_vec[i] = ep.uccl_accept(i, &remote_vdevs[i], remote_ip[i],
                                          ep.listen_fd_vec_[i]);
      }
    }

    uint64_t* data_u64;
    data_u64 = reinterpret_cast<uint64_t*>(data[0]);

    size_t sent_bytes = 0;
    std::vector<uint64_t> rtts;
    auto start_bw_mea = std::chrono::high_resolution_clock::now();

    std::deque<PollCtx*> poll_ctxs;
    PollCtx* last_ctx = nullptr;
    uint32_t inflight_msgs[kNumVdevices] = {0};

    // Synchronization barrier: Wait for server to enter uccl_recv() before flooding.
    // This prevents overwhelming the server before it's ready to receive.
    // 1 second provides robust margin (server takes ~110ms in logs).
    LOG(INFO) << "[Client] Connected, waiting for server to be ready...";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    LOG(INFO) << "[Client] Starting send loop for " << FLAGS_iterations << " iterations...";

    for (size_t i = 0; i < FLAGS_iterations;) {
      send_len = kTestMsgSize;
      if (FLAGS_rand) send_len = distribution(generator);

      if (FLAGS_verify) {
        auto host_data_u64 = get_host_ptr(data_u64, send_len);
        for (int j = 0; j < send_len / sizeof(uint64_t); j++) {
          host_data_u64[j] = (uint64_t)i * (uint64_t)j;
        }
        cudaMemcpy(data_u64, host_data_u64, send_len, cudaMemcpyHostToDevice);
      }

      switch (test_type) {
        case kBasic: {
          TscTimer timer;
          timer.start();
          ep.uccl_send(conn_id, data[0], send_len, &mh[0]);
          timer.stop();
          rtts.push_back(timer.avg_usec(freq_ghz));
          sent_bytes += send_len;
          
          // Optional pacing delay to prevent buffer pool exhaustion in stress tests
          if (FLAGS_delay_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(FLAGS_delay_us));
          }
          
          if (i % 100 == 0) {
            LOG(INFO) << "[Client] Completed iteration " << i 
                     << ", sent " << send_len << " bytes";
          }
          
          i++;
          break;
        }
        case kAsync: {
          std::vector<PollCtx*> poll_ctxs;
          size_t step_size = send_len / kMaxInflight + 1;
          for (int j = 0; j < kMaxInflight; j++) {
            auto iter_len = std::min(step_size, send_len - j * step_size);
            auto* iter_data = data[0] + j * step_size;

            PollCtx* poll_ctx;
            poll_ctx = ep.uccl_send_async(conn_id, iter_data, iter_len, &mh[0]);
            poll_ctx->timestamp = rdtsc();
            poll_ctxs.push_back(poll_ctx);
          }
          for (auto poll_ctx : poll_ctxs) {
            auto async_start = poll_ctx->timestamp;
            // after a success poll, poll_ctx is freed
            ep.uccl_poll(poll_ctx);
            rtts.push_back(to_usec(rdtsc() - async_start, freq_ghz));
          }
          sent_bytes += send_len;
          i++;
          break;
        }
        case kPingpong: {
          PollCtx *poll_ctx1, *poll_ctx2;
          TscTimer timer;
          timer.start();
          poll_ctx1 = ep.uccl_send_async(conn_id, data[0], send_len, &mh[0]);
          poll_ctx2 = ep.uccl_recv_async(conn_id, data2[0], &recv_len, &mh2[0]);
          ep.uccl_poll(poll_ctx1);
          ep.uccl_poll(poll_ctx2);
          timer.stop();
          rtts.push_back(timer.avg_usec(freq_ghz));
          sent_bytes += send_len * 2;
          i += 1;
          break;
        }
        case kMt: {
          TscTimer timer;
          timer.start();
          std::thread t1([&ep, conn_id, data, send_len, mh]() mutable {
            PollCtx* poll_ctx =
                ep.uccl_send_async(conn_id, data[0], send_len, &mh[0]);
            ep.uccl_poll(poll_ctx);
          });
          std::thread t2([&ep, conn_id, data2, &recv_len, mh2]() mutable {
            PollCtx* poll_ctx =
                ep.uccl_recv_async(conn_id, data2[0], &recv_len, &mh2[0]);
            ep.uccl_poll(poll_ctx);
          });
          t1.join();
          t2.join();
          timer.stop();
          rtts.push_back(timer.avg_usec(freq_ghz));
          sent_bytes += send_len * 2;
          i += 1;
          break;
        }
        case kMc: {
          TscTimer timer;
          timer.start();
          for (int j = 0; j < kNumConns; j++) {
            auto* poll_ctx =
                ep.uccl_send_async(conn_id_vec[j], data[0], send_len, &mh[0]);
            poll_ctxs.push_back(poll_ctx);
          }
          while (!poll_ctxs.empty()) {
            auto* poll_ctx = poll_ctxs.front();
            ep.uccl_poll(poll_ctx);
            poll_ctxs.pop_front();
          }
          timer.stop();
          rtts.push_back(timer.avg_usec(freq_ghz));
          sent_bytes += send_len * 2;
          i += 1;
          break;
        }
        case kMq: {
          for (int j = 0; j < kNumVdevices; j++) {
            while (inflight_msgs[j] < kMaxInflight) {
              auto& __conn_id = conn_id_vec[j];
              auto poll_ctx =
                  ep.uccl_send_async(__conn_id, data[j], send_len, &mh[j]);
              poll_ctx->timestamp = rdtsc();
              poll_ctxs.push_back(poll_ctx);
              inflight_msgs[j]++;
            }
          }
          auto inflights = poll_ctxs.size();
          for (int j = 0; j < inflights; j++) {
            auto poll_ctx = poll_ctxs.front();
            poll_ctxs.pop_front();
            auto async_start = poll_ctx->timestamp;
            auto vdev_idx = poll_ctx->engine_idx / kNumEnginesPerVdev;
            if (ep.uccl_poll_once(poll_ctx)) {
              rtts.push_back(to_usec(rdtsc() - async_start, freq_ghz));
              sent_bytes += send_len;
              i++;
              inflight_msgs[vdev_idx]--;
            } else {
              poll_ctxs.push_back(poll_ctx);
            }
          }
          break;
        }
        case kBiMq: {
          for (int j = 0; j < kNumVdevices; j++) {
            while (inflight_msgs[j] < kMaxInflight) {
              auto& __conn_id = conn_id_vec[j];
              auto* poll_ctx =
                  (j % 2 == 0)
                      ? ep.uccl_send_async(__conn_id, data[j], send_len, &mh[j])
                      : ep.uccl_recv_multi_async(__conn_id, (void**)&(data[j]),
                                                 &recv_len, (Mhandle**)&mh[j],
                                                 1);
              poll_ctx->timestamp = rdtsc();
              poll_ctxs.push_back(poll_ctx);
              inflight_msgs[j]++;
            }
          }
          auto inflights = poll_ctxs.size();
          for (int j = 0; j < inflights; j++) {
            auto poll_ctx = poll_ctxs.front();
            auto async_start = poll_ctx->timestamp;
            auto vdev_idx = poll_ctx->engine_idx / kNumEnginesPerVdev;
            poll_ctxs.pop_front();
            if (ep.uccl_poll_once(poll_ctx)) {
              rtts.push_back(to_usec(rdtsc() - async_start, freq_ghz));
              sent_bytes += send_len;
              i++;
              inflight_msgs[vdev_idx]--;
            } else {
              poll_ctxs.push_back(poll_ctx);
            }
          }
          CHECK(send_len == recv_len)
              << "send_len: " << send_len << ", recv_len: " << recv_len;
          break;
        }
        case kTput: {
          // Chunked send with proper client-server coordination
          // Server signals when ready, client waits before sending
          size_t total_size = send_len;
          size_t offset = 0;
          std::vector<PollCtx*> msg_ctxs;
          
          // Wait for server to be ready
          char ready_signal;
          CHECK(recv(oob_sock, &ready_signal, 1, 0) == 1) << "Failed to receive ready signal";
          
          while (offset < total_size) {
            size_t chunk_len = std::min(kChunkSize, total_size - offset);
            
            auto* poll_ctx = ep.uccl_send_async(conn_id, data[0] + offset, chunk_len, &mh[0]);
            poll_ctx->timestamp = rdtsc();
            msg_ctxs.push_back(poll_ctx);
            
            offset += chunk_len;
          }
          
          // Poll all chunks
          for (auto* ctx : msg_ctxs) {
            auto async_start = ctx->timestamp;
            ep.uccl_poll(ctx);
            rtts.push_back(to_usec(rdtsc() - async_start, freq_ghz));
          }
          sent_bytes += total_size;
          i++;
          break;
        }
        default:
          break;
      }

      if ((i + 1) % kReportIters == 0) {
        auto end_bw_mea = std::chrono::high_resolution_clock::now();

        auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>(
            end_bw_mea - start_bw_mea);

        if (duration_sec < kReportIntervalSec) continue;

        // Clear to avoid Percentile() taking too much time.
        if (rtts.size() > 100000) {
          rtts.assign(rtts.end() - 100000, rtts.end());
        }

        auto duaration_usec =
            std::chrono::duration_cast<std::chrono::microseconds>(end_bw_mea -
                                                                  start_bw_mea)
                .count();

        uint64_t med_latency, tail_latency;
        med_latency = Percentile(rtts, 50);
        tail_latency = Percentile(rtts, 99);

        // 24B: 4B FCS + 8B frame delimiter + 12B interframe gap
        auto bw_gbps = sent_bytes *
                       ((EFA_MTU * 1.0 + 24) / (EFA_MTU - kUcclPktHdrLen)) *
                       8.0 / 1000 / 1000 / 1000 / (duaration_usec * 1e-6);
        auto app_bw_gbps =
            sent_bytes * 8.0 / 1000 / 1000 / 1000 / (duaration_usec * 1e-6);
        sent_bytes = 0;

        LOG(INFO) << "Sent " << i + 1 << " messages, med rtt: " << med_latency
                  << " us, tail rtt: " << tail_latency << " us, link bw "
                  << bw_gbps << " Gbps, app bw " << app_bw_gbps << " Gbps";
        start_bw_mea = std::chrono::high_resolution_clock::now();
      }
    }
    
    // For kTput test, drain remaining inflight messages
    if (test_type == kTput) {
      while (!poll_ctxs.empty()) {
        auto* poll_ctx = poll_ctxs.front();
        poll_ctxs.pop_front();
        ep.uccl_poll(poll_ctx);
      }
      LOG(INFO) << "[Client] Completed " << FLAGS_iterations << " sends";
    }
    
    // Final throughput report (for all test types)
    auto end_final = std::chrono::high_resolution_clock::now();
    auto total_duration_usec = std::chrono::duration_cast<std::chrono::microseconds>(
        end_final - start_bw_mea).count();
    
    if (rtts.size() > 0 && total_duration_usec > 0) {
      uint64_t med_latency = Percentile(rtts, 50);
      uint64_t tail_latency = Percentile(rtts, 99);
      
      // Calculate total bytes sent across all iterations
      uint64_t total_bytes = static_cast<uint64_t>(FLAGS_iterations) * kTestMsgSize;
      
      // 24B: 4B FCS + 8B frame delimiter + 12B interframe gap
      auto bw_gbps = total_bytes * ((EFA_MTU * 1.0 + 24) / (EFA_MTU - kUcclPktHdrLen)) * 
                     8.0 / 1000 / 1000 / 1000 / (total_duration_usec * 1e-6);
      auto app_bw_gbps = total_bytes * 8.0 / 1000 / 1000 / 1000 / (total_duration_usec * 1e-6);
      
      LOG(INFO) << "=== FINAL RESULTS ===";
      LOG(INFO) << "Total iterations: " << FLAGS_iterations;
      LOG(INFO) << "Total bytes: " << total_bytes << " (" << total_bytes / (1024*1024) << " MB)";
      LOG(INFO) << "Total time: " << total_duration_usec / 1000000.0 << " seconds";
      LOG(INFO) << "Median RTT: " << med_latency << " us";
      LOG(INFO) << "99th percentile RTT: " << tail_latency << " us";
      LOG(INFO) << "Link bandwidth: " << bw_gbps << " Gbps";
      LOG(INFO) << "Application bandwidth: " << app_bw_gbps << " Gbps";
    }
  } else {
    auto ep = Endpoint();
    ep.initialize_engine_by_gpu_idx(0);
    int const kMaxArraySize = std::max(kNumConns, kNumVdevices);
    ConnID conn_id, conn_id2;
    ConnID conn_id_vec[kMaxArraySize];
    std::string remote_ip[kMaxArraySize];
    int remote_vdevs[kMaxArraySize];

    for (int i = 0; i < kMaxArraySize; i++) ep.uccl_listen();

    // Exchange listen ports with client via out-of-band TCP connection
    std::vector<uint16_t> remote_listen_ports(kMaxArraySize);
    LOG(INFO) << "[Server] Listening for OOB connection on port " << FLAGS_oobport;
    LOG(INFO) << "[Server] My listen ports: " << ep.listen_port_vec_[0]
              << " (use this for reference, but client will get all ports via OOB)";
    listen_accept_exchange(FLAGS_oobport, ep.listen_port_vec_.data(),
                           ep.listen_port_vec_.size() * sizeof(uint16_t),
                           remote_listen_ports.data(),
                           kMaxArraySize * sizeof(uint16_t));
    LOG(INFO) << "[Server] Exchanged ports with client, preparing CUDA buffers...";

    // Allocate and register CUDA memory BEFORE accepting connections.
    // This prevents buffer exhaustion during the ~100ms CUDA setup delay.
    int send_len = kTestMsgSize, recv_len = kTestMsgSize;
    // Use synchronous mode (pipeline depth 1) to avoid buffer reuse issues
    // With async APIs, buffers may still be in use even after poll() returns
    constexpr int kPipelineDepth = 1;
    constexpr int kNumBuffers = 8;  // Allocate extra buffers for safety
    uint8_t *data[kNumBuffers], *data2[kNumBuffers];
    Mhandle mh[kNumBuffers], mh2[kNumBuffers];

    auto gpu_idx = 0;  // Use single GPU for all buffers
    auto dev_idx = get_dev_idx_by_gpu_idx(0);
    cudaSetDevice(gpu_idx);
    auto* dev = EFAFactory::GetEFADevice(dev_idx);

    for (int i = 0; i < kNumBuffers; i++) {

#ifdef INTEL_RDMA_NIC
      cudaMallocManaged(&data[i], kTestMsgSize, cudaMemAttachGlobal);
      mh[i].mr =
          ibv_reg_mr(dev->pd, data[i], kTestMsgSize,
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
      cudaMallocManaged(&data2[i], kTestMsgSize, cudaMemAttachGlobal);
      mh2[i].mr =
          ibv_reg_mr(dev->pd, data2[i], kTestMsgSize,
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
#else
      cudaMalloc(&data[i], kTestMsgSize);
      mh[i].mr =
          ibv_reg_mr(dev->pd, data[i], kTestMsgSize, IBV_ACCESS_LOCAL_WRITE);
      cudaMalloc(&data2[i], kTestMsgSize);
      mh2[i].mr =
          ibv_reg_mr(dev->pd, data2[i], kTestMsgSize, IBV_ACCESS_LOCAL_WRITE);
#endif
    }
    cudaSetDevice(0);

    LOG(INFO) << "[Server] CUDA buffers ready, waiting for connection...";

    // Create persistent OOB socket for coordination (only for kTput test)
    if (test_type == kTput) {
      LOG(INFO) << "[Server] Creating persistent OOB socket for flow control...";
      oob_sock = create_oob_server_socket(FLAGS_oobport);
      LOG(INFO) << "[Server] OOB connection established";
    }

    conn_id =
        ep.uccl_accept(0, &remote_vdevs[0], remote_ip[0], ep.listen_fd_vec_[0]);
    
    if (test_type == kMc) {
      conn_id_vec[0] = conn_id;
      for (int i = 1; i < kNumConns; i++) {
        conn_id_vec[i] = ep.uccl_accept(0, &remote_vdevs[i], remote_ip[i],
                                        ep.listen_fd_vec_[0]);
      }
    } else if (test_type == kMq) {
      conn_id_vec[0] = conn_id;
      for (int i = 1; i < kNumVdevices; i++) {
        conn_id_vec[i] = ep.uccl_accept(i, &remote_vdevs[i], remote_ip[i],
                                        ep.listen_fd_vec_[i]);
      }
    } else if (test_type == kBiMq) {
      conn_id_vec[0] = conn_id;
      for (int i = 1; i < kNumVdevices; i++) {
        if (i % 2 == 0)
          conn_id_vec[i] = ep.uccl_accept(i, &remote_vdevs[i], remote_ip[i],
                                          ep.listen_fd_vec_[i]);
        else
          conn_id_vec[i] =
              ep.uccl_connect(i, i, FLAGS_clientip, remote_listen_ports[i]);
      }
    }

    uint64_t* data_u64;
    data_u64 = reinterpret_cast<uint64_t*>(data[0]);
    auto start = std::chrono::high_resolution_clock::now();

    std::deque<PollCtx*> poll_ctxs;
    PollCtx* last_ctx = nullptr;
    uint32_t inflight_msgs[kNumVdevices] = {0};

    // Pre-post receive buffers BEFORE signaling client, to prevent buffer exhaustion.
    // For synchronous blocking mode, no need to pre-post
    LOG(INFO) << "[Server] Ready to receive (synchronous blocking mode)";

    // Synchronization barrier: Signal to client that server is ready to receive.
    LOG(INFO) << "[Server] Signaling client via OOB...";
    char ready_signal = 'R';
    LOG(INFO) << "[Server] Entering receive loop for " << FLAGS_iterations << " iterations...";

    for (size_t i = 0; i < FLAGS_iterations;) {
      send_len = kTestMsgSize;
      if (FLAGS_rand) send_len = distribution(generator);

      switch (test_type) {
        case kBasic: {
          // Use busypoll mode (true) to allow engine thread to keep processing
          // and freeing buffers, preventing exhaustion in tight loops
          ep.uccl_recv(conn_id, data[0], &recv_len, &mh[0], true);
          
          // Add delay to allow engine thread to process ACKs and free buffers
          if (FLAGS_delay_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(FLAGS_delay_us));
          }
          
          if (i % 100 == 0) {
            LOG(INFO) << "[Server] Completed iteration " << i
                     << ", received " << recv_len << " bytes";
          }
          
          i++;
          break;
        }
        case kAsync: {
          size_t step_size = send_len / kMaxInflight + 1;
          int recv_lens[kMaxInflight] = {0};
          std::vector<PollCtx*> poll_ctxs;
          for (int j = 0; j < kMaxInflight; j++) {
            auto iter_len = std::min(step_size, send_len - j * step_size);
            auto* iter_data = data[0] + j * step_size;

            PollCtx* poll_ctx;
            poll_ctx =
                ep.uccl_recv_async(conn_id, iter_data, &recv_lens[j], &mh[0]);
            poll_ctxs.push_back(poll_ctx);
          }
          for (auto poll_ctx : poll_ctxs) {
            ep.uccl_poll(poll_ctx);
          }
          recv_len = 0;
          for (auto len : recv_lens) {
            recv_len += len;
          }
          i++;
          break;
        }
        case kPingpong: {
          PollCtx *poll_ctx1, *poll_ctx2;
          poll_ctx1 = ep.uccl_recv_async(conn_id, data[0], &recv_len, &mh[0]);
          poll_ctx2 = ep.uccl_send_async(conn_id, data2[0], send_len, &mh2[0]);
          ep.uccl_poll(poll_ctx1);
          ep.uccl_poll(poll_ctx2);
          i += 1;
          break;
        }
        case kMt: {
          std::thread t1([&ep, conn_id, data, &recv_len, mh]() mutable {
            PollCtx* poll_ctx =
                ep.uccl_recv_async(conn_id, data[0], &recv_len, &mh[0]);
            ep.uccl_poll(poll_ctx);
          });
          std::thread t2([&ep, conn_id, data2, send_len, mh2]() mutable {
            PollCtx* poll_ctx =
                ep.uccl_send_async(conn_id, data2[0], send_len, &mh2[0]);
            ep.uccl_poll(poll_ctx);
          });
          t1.join();
          t2.join();
          i += 1;
          break;
        }
        case kMc: {
          for (int j = 0; j < kNumConns; j++) {
            auto* poll_ctx =
                ep.uccl_recv_async(conn_id_vec[j], data[0], &recv_len, &mh[0]);
            poll_ctxs.push_back(poll_ctx);
          }
          while (!poll_ctxs.empty()) {
            auto* poll_ctx = poll_ctxs.front();
            ep.uccl_poll(poll_ctx);
            poll_ctxs.pop_front();
          }
          i += 1;
          break;
        }
        case kMq: {
          for (int j = 0; j < kNumVdevices; j++) {
            while (inflight_msgs[j] < kMaxInflight) {
              auto& __conn_id = conn_id_vec[j];
              auto poll_ctx =
                  ep.uccl_recv_async(__conn_id, data[j], &recv_len, &mh[j]);
              poll_ctxs.push_back(poll_ctx);
              inflight_msgs[j]++;
            }
          }
          auto inflights = poll_ctxs.size();
          for (int j = 0; j < inflights; j++) {
            auto poll_ctx = poll_ctxs.front();
            poll_ctxs.pop_front();
            auto vdev_idx = poll_ctx->engine_idx / kNumEnginesPerVdev;
            if (ep.uccl_poll_once(poll_ctx)) {
              inflight_msgs[vdev_idx]--;
              i++;
            } else {
              poll_ctxs.push_back(poll_ctx);
            }
          }
          break;
        }
        case kBiMq: {
          for (int j = 0; j < kNumVdevices; j++) {
            while (inflight_msgs[j] < kMaxInflight) {
              auto& __conn_id = conn_id_vec[j];
              auto* poll_ctx = (j % 2 == 0)
                                   ? ep.uccl_recv_async(__conn_id, data[j],
                                                        &recv_len, &mh[j])
                                   : ep.uccl_send_async(__conn_id, data[j],
                                                        send_len, &mh[j]);
              poll_ctxs.push_back(poll_ctx);
              inflight_msgs[j]++;
            }
          }
          auto inflights = poll_ctxs.size();
          for (int j = 0; j < inflights; j++) {
            auto poll_ctx = poll_ctxs.front();
            auto vdev_idx = poll_ctx->engine_idx / kNumEnginesPerVdev;
            poll_ctxs.pop_front();
            if (ep.uccl_poll_once(poll_ctx)) {
              inflight_msgs[vdev_idx]--;
              i++;
            } else {
              poll_ctxs.push_back(poll_ctx);
            }
          }
          CHECK(send_len == recv_len)
              << "send_len: " << send_len << ", recv_len: " << recv_len;
          break;
        }
        case kTput: {
          // Signal client we're ready for this message
          char ready_signal = 'R';
          CHECK(send(oob_sock, &ready_signal, 1, 0) == 1) << "Failed to send ready signal";
          
          // Chunked receive - post all receives for this message
          size_t total_size = send_len;
          size_t num_chunks = (total_size + kChunkSize - 1) / kChunkSize;
          size_t offset = 0;
          
          std::vector<PollCtx*> msg_ctxs;
          std::vector<int> recv_lens(num_chunks, 0);
          
          for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
            auto* poll_ctx = ep.uccl_recv_async(conn_id, data[0] + offset,
                                                &recv_lens[chunk_idx], &mh[0]);
            msg_ctxs.push_back(poll_ctx);
            offset += kChunkSize;
          }
          
          // Poll all chunks
          recv_len = 0;
          for (auto* ctx : msg_ctxs) {
            ep.uccl_poll(ctx);
          }
          for (int len : recv_lens) {
            recv_len += len;
          }
          
          i++;
          break;
        }
        default:
          break;
      }

      if ((i + 1) % kReportIters == 0) {
        auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - start);
        if (duration_sec < kReportIntervalSec) continue;

        LOG(INFO) << "Received " << i + 1 << " messages";

        start = std::chrono::high_resolution_clock::now();
      }

      if (FLAGS_verify) {
        auto host_data_u64 = get_host_ptr(data_u64, send_len);

        bool data_mismatch = false;
        auto expected_len = FLAGS_rand ? send_len : kTestMsgSize;
        if (recv_len != expected_len) {
          LOG(ERROR) << "Received message size mismatches, expected "
                     << expected_len << ", received " << recv_len;
          data_mismatch = true;
        }
        for (int j = 0; j < recv_len / sizeof(uint64_t); j++) {
          if (host_data_u64[j] != (uint64_t)i * (uint64_t)j) {
            data_mismatch = true;
            LOG_EVERY_N(ERROR, 1000)
                << "Data mismatch at index " << j * sizeof(uint64_t)
                << ", expected " << (uint64_t)i * (uint64_t)j << ", received "
                << host_data_u64[j];
          }
        }
        CHECK(!data_mismatch) << "Data mismatch at iter " << i;
        memset(host_data_u64, 0, recv_len);
        cudaMemcpy(data_u64, host_data_u64, send_len, cudaMemcpyHostToDevice);
      }
    }
  }

  // Close OOB socket if it was opened
  if (test_type == kTput && oob_sock >= 0) {
    close(oob_sock);
  }

  return 0;
}