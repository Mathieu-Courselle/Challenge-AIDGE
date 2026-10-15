typedef float float32x4_t __attribute__((vector_size(16)));
static inline float32x4_t vdupq_n_f32(float x){ float32x4_t r={x,x,x,x}; return r; }
static inline float32x4_t vld1q_f32(const float*p){ float32x4_t r; __builtin_memcpy(&r,p,16); return r; }
static inline void vst1q_f32(float*p,float32x4_t v){ __builtin_memcpy(p,&v,16); }
static inline float32x4_t vmlaq_n_f32(float32x4_t a,float32x4_t b,float c){ return a+b*c; }
