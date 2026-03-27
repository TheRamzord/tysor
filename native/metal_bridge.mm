#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

namespace {

thread_local std::string g_last_error;

void SetError(const std::string& message) { g_last_error = message; }

struct Context {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
};

struct Buffer {
  id<MTLBuffer> handle;
  std::size_t count;
};

id<MTLComputePipelineState> RequirePipeline(Context* context, const char* kernel_name) {
  auto it = context->pipelines.find(kernel_name);
  if (it != context->pipelines.end()) {
    return it->second;
  }
  NSString* name = [NSString stringWithUTF8String:kernel_name];
  id<MTLFunction> function = [context->library newFunctionWithName:name];
  if (!function) {
    throw std::runtime_error("Metal bridge could not resolve kernel function");
  }
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline =
      [context->device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline) {
    std::string message = error ? [[error localizedDescription] UTF8String] : "unknown error";
    throw std::runtime_error("Metal bridge could not create compute pipeline: " + message);
  }
  context->pipelines.emplace(kernel_name, pipeline);
  return pipeline;
}

id<MTLBuffer> MakeScalarBuffer(id<MTLDevice> device, uint32_t value) {
  return [device newBufferWithBytes:&value length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
}

void Dispatch1D(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, uint32_t count) {
  MTLSize threadsPerGrid = MTLSizeMake(count, 1, 1);
  NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
  if (width == 0) width = 1;
  MTLSize threadsPerThreadgroup = MTLSizeMake(std::min<NSUInteger>(width, count == 0 ? 1u : count), 1, 1);
  [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
}

void Dispatch2D(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
                uint32_t width, uint32_t height) {
  MTLSize threadsPerGrid = MTLSizeMake(width, height, 1);
  NSUInteger tw = std::min<NSUInteger>(pipeline.threadExecutionWidth ? pipeline.threadExecutionWidth : 8, width);
  NSUInteger th = std::max<NSUInteger>(1, std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup / tw, height));
  MTLSize threadsPerThreadgroup = MTLSizeMake(tw == 0 ? 1 : tw, th == 0 ? 1 : th, 1);
  [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
}

template <typename ConfigureFn, typename DispatchFn>
bool WithEncoder(Context* context, const char* kernel_name, ConfigureFn configure, DispatchFn dispatch) {
  @autoreleasepool {
    try {
      id<MTLComputePipelineState> pipeline = RequirePipeline(context, kernel_name);
      id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
      if (!command_buffer) {
        throw std::runtime_error("Metal bridge could not create command buffer");
      }
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (!encoder) {
        throw std::runtime_error("Metal bridge could not create compute encoder");
      }
      [encoder setComputePipelineState:pipeline];
      configure(encoder);
      dispatch(encoder, pipeline);
      [encoder endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      return true;
    } catch (const std::exception& err) {
      SetError(err.what());
      return false;
    }
  }
}

}  // namespace

extern "C" {

const char* tysor_metal_last_error(void) { return g_last_error.c_str(); }

void* tysor_metal_context_new(const char* source) {
  @autoreleasepool {
    try {
      auto context = std::make_unique<Context>();
      context->device = MTLCreateSystemDefaultDevice();
      if (!context->device) {
        throw std::runtime_error("Metal bridge could not create a default device");
      }
      context->queue = [context->device newCommandQueue];
      if (!context->queue) {
        throw std::runtime_error("Metal bridge could not create command queue");
      }
      NSString* metal_source = [NSString stringWithUTF8String:source];
      NSError* error = nil;
      context->library = [context->device newLibraryWithSource:metal_source options:nil error:&error];
      if (!context->library) {
        std::string message = error ? [[error localizedDescription] UTF8String] : "unknown error";
        throw std::runtime_error("Metal bridge failed to compile source: " + message);
      }
      return context.release();
    } catch (const std::exception& err) {
      SetError(err.what());
      return nullptr;
    }
  }
}

void tysor_metal_context_free(void* context) { delete static_cast<Context*>(context); }

void* tysor_metal_buffer_new_with_data(void* context, const float* data, std::size_t count) {
  @autoreleasepool {
    try {
      auto* metal = static_cast<Context*>(context);
      std::size_t bytes = count * sizeof(float);
      id<MTLBuffer> buffer =
          [metal->device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
      if (!buffer) {
        throw std::runtime_error("Metal bridge failed to create upload buffer");
      }
      return new Buffer{buffer, count};
    } catch (const std::exception& err) {
      SetError(err.what());
      return nullptr;
    }
  }
}

void* tysor_metal_buffer_new_zeroed(void* context, std::size_t count) {
  @autoreleasepool {
    try {
      auto* metal = static_cast<Context*>(context);
      id<MTLBuffer> buffer =
          [metal->device newBufferWithLength:count * sizeof(float) options:MTLResourceStorageModeShared];
      if (!buffer) {
        throw std::runtime_error("Metal bridge failed to allocate device buffer");
      }
      std::memset([buffer contents], 0, count * sizeof(float));
      return new Buffer{buffer, count};
    } catch (const std::exception& err) {
      SetError(err.what());
      return nullptr;
    }
  }
}

void tysor_metal_buffer_free(void* buffer) { delete static_cast<Buffer*>(buffer); }

bool tysor_metal_buffer_read(void* buffer, float* out_data, std::size_t count) {
  try {
    auto* metal_buffer = static_cast<Buffer*>(buffer);
    if (count > metal_buffer->count) {
      throw std::runtime_error("Metal bridge read exceeds buffer length");
    }
    std::memcpy(out_data, [metal_buffer->handle contents], count * sizeof(float));
    return true;
  } catch (const std::exception& err) {
    SetError(err.what());
    return false;
  }
}

bool tysor_metal_dispatch_matmul(void* context, const char* kernel_name, void* lhs, void* rhs, void* out,
                                 uint32_t m, uint32_t n, uint32_t k) {
  auto* metal = static_cast<Context*>(context);
  auto* lhs_buffer = static_cast<Buffer*>(lhs);
  auto* rhs_buffer = static_cast<Buffer*>(rhs);
  auto* out_buffer = static_cast<Buffer*>(out);
  return WithEncoder(
      metal, kernel_name,
      [&](id<MTLComputeCommandEncoder> encoder) {
        id<MTLBuffer> m_buf = MakeScalarBuffer(metal->device, m);
        id<MTLBuffer> n_buf = MakeScalarBuffer(metal->device, n);
        id<MTLBuffer> k_buf = MakeScalarBuffer(metal->device, k);
        [encoder setBuffer:lhs_buffer->handle offset:0 atIndex:0];
        [encoder setBuffer:rhs_buffer->handle offset:0 atIndex:1];
        [encoder setBuffer:out_buffer->handle offset:0 atIndex:2];
        [encoder setBuffer:m_buf offset:0 atIndex:3];
        [encoder setBuffer:n_buf offset:0 atIndex:4];
        [encoder setBuffer:k_buf offset:0 atIndex:5];
      },
      [&](id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline) {
        Dispatch2D(encoder, pipeline, n, m);
      });
}

bool tysor_metal_dispatch_unary(void* context, const char* kernel_name, void* input, void* out, uint32_t count) {
  auto* metal = static_cast<Context*>(context);
  auto* in_buffer = static_cast<Buffer*>(input);
  auto* out_buffer = static_cast<Buffer*>(out);
  return WithEncoder(
      metal, kernel_name,
      [&](id<MTLComputeCommandEncoder> encoder) {
        id<MTLBuffer> count_buf = MakeScalarBuffer(metal->device, count);
        [encoder setBuffer:in_buffer->handle offset:0 atIndex:0];
        [encoder setBuffer:out_buffer->handle offset:0 atIndex:1];
        [encoder setBuffer:count_buf offset:0 atIndex:2];
      },
      [&](id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline) {
        Dispatch1D(encoder, pipeline, count);
      });
}

bool tysor_metal_dispatch_binary_tt(void* context, const char* kernel_name, void* lhs, void* rhs, void* out, uint32_t count) {
  auto* metal = static_cast<Context*>(context);
  auto* lhs_buffer = static_cast<Buffer*>(lhs);
  auto* rhs_buffer = static_cast<Buffer*>(rhs);
  auto* out_buffer = static_cast<Buffer*>(out);
  return WithEncoder(
      metal, kernel_name,
      [&](id<MTLComputeCommandEncoder> encoder) {
        id<MTLBuffer> count_buf = MakeScalarBuffer(metal->device, count);
        [encoder setBuffer:lhs_buffer->handle offset:0 atIndex:0];
        [encoder setBuffer:rhs_buffer->handle offset:0 atIndex:1];
        [encoder setBuffer:out_buffer->handle offset:0 atIndex:2];
        [encoder setBuffer:count_buf offset:0 atIndex:3];
      },
      [&](id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline) {
        Dispatch1D(encoder, pipeline, count);
      });
}

bool tysor_metal_dispatch_binary_ts(void* context, const char* kernel_name, void* input, void* out, uint32_t count) {
  auto* metal = static_cast<Context*>(context);
  auto* in_buffer = static_cast<Buffer*>(input);
  auto* out_buffer = static_cast<Buffer*>(out);
  return WithEncoder(
      metal, kernel_name,
      [&](id<MTLComputeCommandEncoder> encoder) {
        id<MTLBuffer> count_buf = MakeScalarBuffer(metal->device, count);
        [encoder setBuffer:in_buffer->handle offset:0 atIndex:0];
        [encoder setBuffer:out_buffer->handle offset:0 atIndex:1];
        [encoder setBuffer:count_buf offset:0 atIndex:2];
      },
      [&](id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline) {
        Dispatch1D(encoder, pipeline, count);
      });
}

bool tysor_metal_dispatch_binary_st(void* context, const char* kernel_name, void* input, void* out, uint32_t count) {
  return tysor_metal_dispatch_binary_ts(context, kernel_name, input, out, count);
}

bool tysor_metal_dispatch_rms_norm(void* context, const char* kernel_name, void* input, void* out,
                                   uint32_t rows, uint32_t width) {
  auto* metal = static_cast<Context*>(context);
  auto* in_buffer = static_cast<Buffer*>(input);
  auto* out_buffer = static_cast<Buffer*>(out);
  return WithEncoder(
      metal, kernel_name,
      [&](id<MTLComputeCommandEncoder> encoder) {
        id<MTLBuffer> rows_buf = MakeScalarBuffer(metal->device, rows);
        id<MTLBuffer> width_buf = MakeScalarBuffer(metal->device, width);
        [encoder setBuffer:in_buffer->handle offset:0 atIndex:0];
        [encoder setBuffer:out_buffer->handle offset:0 atIndex:1];
        [encoder setBuffer:rows_buf offset:0 atIndex:2];
        [encoder setBuffer:width_buf offset:0 atIndex:3];
      },
      [&](id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline) {
        Dispatch1D(encoder, pipeline, rows);
      });
}

bool tysor_metal_dispatch_softmax(void* context, const char* kernel_name, void* input, void* out,
                                  uint32_t rows, uint32_t width) {
  return tysor_metal_dispatch_rms_norm(context, kernel_name, input, out, rows, width);
}

bool tysor_metal_dispatch_cross_entropy(void* context, const char* kernel_name, void* logits, void* target, void* out,
                                        uint32_t rows, uint32_t width) {
  auto* metal = static_cast<Context*>(context);
  auto* logits_buffer = static_cast<Buffer*>(logits);
  auto* target_buffer = static_cast<Buffer*>(target);
  auto* out_buffer = static_cast<Buffer*>(out);
  return WithEncoder(
      metal, kernel_name,
      [&](id<MTLComputeCommandEncoder> encoder) {
        id<MTLBuffer> rows_buf = MakeScalarBuffer(metal->device, rows);
        id<MTLBuffer> width_buf = MakeScalarBuffer(metal->device, width);
        [encoder setBuffer:logits_buffer->handle offset:0 atIndex:0];
        [encoder setBuffer:target_buffer->handle offset:0 atIndex:1];
        [encoder setBuffer:out_buffer->handle offset:0 atIndex:2];
        [encoder setBuffer:rows_buf offset:0 atIndex:3];
        [encoder setBuffer:width_buf offset:0 atIndex:4];
      },
      [&](id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline) {
        Dispatch1D(encoder, pipeline, rows);
      });
}

bool tysor_metal_dispatch_linear(void* context, const char* kernel_name, void* input, void* weight, void* bias,
                                 void* out, uint32_t m, uint32_t n, uint32_t k, bool with_bias) {
  auto* metal = static_cast<Context*>(context);
  auto* in_buffer = static_cast<Buffer*>(input);
  auto* weight_buffer = static_cast<Buffer*>(weight);
  auto* bias_buffer = static_cast<Buffer*>(bias);
  auto* out_buffer = static_cast<Buffer*>(out);
  return WithEncoder(
      metal, kernel_name,
      [&](id<MTLComputeCommandEncoder> encoder) {
        id<MTLBuffer> m_buf = MakeScalarBuffer(metal->device, m);
        id<MTLBuffer> n_buf = MakeScalarBuffer(metal->device, n);
        id<MTLBuffer> k_buf = MakeScalarBuffer(metal->device, k);
        [encoder setBuffer:in_buffer->handle offset:0 atIndex:0];
        [encoder setBuffer:weight_buffer->handle offset:0 atIndex:1];
        if (with_bias) {
          [encoder setBuffer:bias_buffer->handle offset:0 atIndex:2];
          [encoder setBuffer:out_buffer->handle offset:0 atIndex:3];
          [encoder setBuffer:m_buf offset:0 atIndex:4];
          [encoder setBuffer:n_buf offset:0 atIndex:5];
          [encoder setBuffer:k_buf offset:0 atIndex:6];
        } else {
          [encoder setBuffer:out_buffer->handle offset:0 atIndex:2];
          [encoder setBuffer:m_buf offset:0 atIndex:3];
          [encoder setBuffer:n_buf offset:0 atIndex:4];
          [encoder setBuffer:k_buf offset:0 atIndex:5];
        }
      },
      [&](id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline) {
        Dispatch2D(encoder, pipeline, n, m);
      });
}

bool tysor_metal_dispatch_embedding(void* context, const char* kernel_name, void* indices, void* weight, void* out,
                                    uint32_t index_count, uint32_t embedding_dim) {
  auto* metal = static_cast<Context*>(context);
  auto* indices_buffer = static_cast<Buffer*>(indices);
  auto* weight_buffer = static_cast<Buffer*>(weight);
  auto* out_buffer = static_cast<Buffer*>(out);
  return WithEncoder(
      metal, kernel_name,
      [&](id<MTLComputeCommandEncoder> encoder) {
        id<MTLBuffer> count_buf = MakeScalarBuffer(metal->device, index_count);
        id<MTLBuffer> dim_buf = MakeScalarBuffer(metal->device, embedding_dim);
        [encoder setBuffer:indices_buffer->handle offset:0 atIndex:0];
        [encoder setBuffer:weight_buffer->handle offset:0 atIndex:1];
        [encoder setBuffer:out_buffer->handle offset:0 atIndex:2];
        [encoder setBuffer:count_buf offset:0 atIndex:3];
        [encoder setBuffer:dim_buf offset:0 atIndex:4];
      },
      [&](id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline) {
        Dispatch2D(encoder, pipeline, embedding_dim, index_count);
      });
}

bool tysor_metal_dispatch_repeat_kv(void* context, const char* kernel_name, void* input, void* out,
                                    uint32_t outer, uint32_t out_heads, uint32_t inner, uint32_t repeats) {
  auto* metal = static_cast<Context*>(context);
  auto* in_buffer = static_cast<Buffer*>(input);
  auto* out_buffer = static_cast<Buffer*>(out);
  return WithEncoder(
      metal, kernel_name,
      [&](id<MTLComputeCommandEncoder> encoder) {
        id<MTLBuffer> outer_buf = MakeScalarBuffer(metal->device, outer);
        id<MTLBuffer> heads_buf = MakeScalarBuffer(metal->device, out_heads);
        id<MTLBuffer> inner_buf = MakeScalarBuffer(metal->device, inner);
        id<MTLBuffer> repeats_buf = MakeScalarBuffer(metal->device, repeats);
        [encoder setBuffer:in_buffer->handle offset:0 atIndex:0];
        [encoder setBuffer:out_buffer->handle offset:0 atIndex:1];
        [encoder setBuffer:outer_buf offset:0 atIndex:2];
        [encoder setBuffer:heads_buf offset:0 atIndex:3];
        [encoder setBuffer:inner_buf offset:0 atIndex:4];
        [encoder setBuffer:repeats_buf offset:0 atIndex:5];
      },
      [&](id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline) {
        Dispatch2D(encoder, pipeline, inner, outer * out_heads);
      });
}

bool tysor_metal_dispatch_causal_mask(void* context, const char* kernel_name, void* input, void* out,
                                      uint32_t outer, uint32_t q, uint32_t k) {
  auto* metal = static_cast<Context*>(context);
  auto* in_buffer = static_cast<Buffer*>(input);
  auto* out_buffer = static_cast<Buffer*>(out);
  return WithEncoder(
      metal, kernel_name,
      [&](id<MTLComputeCommandEncoder> encoder) {
        id<MTLBuffer> outer_buf = MakeScalarBuffer(metal->device, outer);
        id<MTLBuffer> q_buf = MakeScalarBuffer(metal->device, q);
        id<MTLBuffer> k_buf = MakeScalarBuffer(metal->device, k);
        [encoder setBuffer:in_buffer->handle offset:0 atIndex:0];
        [encoder setBuffer:out_buffer->handle offset:0 atIndex:1];
        [encoder setBuffer:outer_buf offset:0 atIndex:2];
        [encoder setBuffer:q_buf offset:0 atIndex:3];
        [encoder setBuffer:k_buf offset:0 atIndex:4];
      },
      [&](id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline) {
        Dispatch2D(encoder, pipeline, k, outer * q);
      });
}

bool tysor_metal_dispatch_rope(void* context, const char* kernel_name, void* input, void* out,
                               uint32_t outer, uint32_t seq_len, uint32_t half_dim) {
  auto* metal = static_cast<Context*>(context);
  auto* in_buffer = static_cast<Buffer*>(input);
  auto* out_buffer = static_cast<Buffer*>(out);
  return WithEncoder(
      metal, kernel_name,
      [&](id<MTLComputeCommandEncoder> encoder) {
        id<MTLBuffer> outer_buf = MakeScalarBuffer(metal->device, outer);
        id<MTLBuffer> seq_buf = MakeScalarBuffer(metal->device, seq_len);
        [encoder setBuffer:in_buffer->handle offset:0 atIndex:0];
        [encoder setBuffer:out_buffer->handle offset:0 atIndex:1];
        [encoder setBuffer:outer_buf offset:0 atIndex:2];
        [encoder setBuffer:seq_buf offset:0 atIndex:3];
      },
      [&](id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline) {
        Dispatch2D(encoder, pipeline, half_dim, outer * seq_len);
      });
}

}  // extern "C"
