// SPH fluid simulation step: density/pressure -> fluid forces -> boundary
// forces -> leapfrog position update (a four-kernel pipeline).
//
// Extracted from HeCBench src/sph-cuda/fluid.cu and common.h.
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5 (BSD-3-Clause).
// All device functions and the four kernels below are upstream code verbatim
// (including the idiosyncratic `xi = (1-x/h)?x<h:0.0;` line in
// boundaryGamma). Only the host harness is new: deterministic lattice
// initialization, fixed parameters, two pipeline steps, text output.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

// ---- upstream structs (common.h, verbatim fields) ---------------------------
struct boundary_particle {
    double3 pos; // position
    double3 n;   // position
};

struct fluid_particle {
    double density;
    double pressure;
    double3 pos;     // position
    double3 v;       // velocity
    double3 v_half;  // half step velocity
    double3 a;       // acceleration
};

struct param {
    double rest_density;
    double mass_particle;
    double spacing_particle;
    double smoothing_radius;
    double g;
    double time_step;
    double alpha;
    double surface_tension;
    double speed_sound;
    int number_particles;
    int number_fluid_particles;
    int number_boundary_particles;
    int number_steps;
    int steps_per_frame;
};

// ---- upstream device code (fluid.cu, verbatim) -------------------------------
__device__
double W(double3 p_pos, double3 q_pos, double h)
{
    double r = sqrt((p_pos.x-q_pos.x)*(p_pos.x-q_pos.x)
                  + (p_pos.y-q_pos.y)*(p_pos.y-q_pos.y)
                  + (p_pos.z-q_pos.z)*(p_pos.z-q_pos.z));
    double C = 1.0/(M_PI*h*h*h);
    double u = r/h;
    double val = 0.0;
    if(u >= 2.0)
        return val;
    else if(u < 1.0 )
        val = 1.0 - (3.0/2.0)*u*u + (3.0/4.0)*u*u*u;
    else if(u >= 1.0 && u < 2.0)
        val = (1.0/4.0) * pow(2.0-u,3.0);

    val *= C;
    return val;
}

// Gradient of B spline kernel
__device__
double del_W(double3 p_pos, double3 q_pos, double h)
{
    double r = sqrt((p_pos.x-q_pos.x)*(p_pos.x-q_pos.x)
                  + (p_pos.y-q_pos.y)*(p_pos.y-q_pos.y)
                  + (p_pos.z-q_pos.z)*(p_pos.z-q_pos.z));
    double C = 1.0/(M_PI * h*h*h);
    double u = r/h;
    double val = 0.0;
    if(u >= 2.0)
        return val;
    else if(u < 1.0 )
        val = -1.0/(h*h) * (3.0 - 9.0/4.0*u);
    else if(u >= 1.0 && u < 2.0)
        val = -3.0/(4.0*h*r) * pow(2.0-u,2.0);

    val *= C;
    return val;
}

__device__
double boundaryGamma(double3 p_pos, double3 k_pos, double3 k_n, double h, double speed_sound)
{
    // Radial distance between p,q
    double r = sqrt((p_pos.x-k_pos.x)*(p_pos.x-k_pos.x)
                  + (p_pos.y-k_pos.y)*(p_pos.y-k_pos.y)
                  + (p_pos.z-k_pos.z)*(p_pos.z-k_pos.z));
    // Distance to p normal to surface particle
    double y = sqrt((p_pos.x-k_pos.x)*(p_pos.x-k_pos.x)*(k_n.x*k_n.x)
                  + (p_pos.y-k_pos.y)*(p_pos.y-k_pos.y)*(k_n.y*k_n.y)
                  + (p_pos.z-k_pos.z)*(p_pos.z-k_pos.z)*(k_n.z*k_n.z));
    // Tangential distance
    double x = r-y;

    double u = y/h;
    double xi = (1-x/h)?x<h:0.0;
    double C = xi*2.0*0.02 * speed_sound * speed_sound / y;
    double val = 0.0;

    if(u > 0.0 && u < 2.0/3.0)
        val = 2.0/3.0;
    else if(u < 1.0 && u > 2.0/3.0 )
        val = (2*u - 3.0/2.0*u*u);
    else if (u < 2.0 && u > 1.0)
        val = 0.5*(2.0-u)*(2.0-u);
    else
        val = 0.0;

    val *= C;

    return val;
}

__device__
double computeDensity(double3 p_pos, double3 p_v, double3 q_pos, double3 q_v,
                      const param *params)
{
    double v_x = (p_v.x - q_v.x);
    double v_y = (p_v.y - q_v.y);
    double v_z = (p_v.z - q_v.z);

    double density = params->mass_particle * del_W(p_pos,q_pos,
                                                   params->smoothing_radius);
    double density_x = density * v_x * (p_pos.x - q_pos.x);
    double density_y = density * v_y * (p_pos.y - q_pos.y);
    double density_z = density * v_z * (p_pos.z - q_pos.z);

    density = (density_x + density_y + density_z)*params->time_step;

    return density;
}

__device__
double computePressure(double p_density, const param *params)
{
    double gam = 7.0;
    double B = params->rest_density * params->speed_sound*params->speed_sound / gam;
    double pressure =  B * (pow((p_density/params->rest_density),gam) - 1.0);

    return pressure;
}

__global__
void updatePressures(fluid_particle *__restrict__ fluid_particles,
                     const param *__restrict__ params)
{
    int num_fluid_particles = params->number_fluid_particles;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_fluid_particles) return;
    double3 p_pos = fluid_particles[i].pos;
    double3 p_v   = fluid_particles[i].v;
    double density = fluid_particles[i].density;

    for(int j=0; j< num_fluid_particles; j++) {
        double3 q_pos = fluid_particles[j].pos;
        double3 q_v   = fluid_particles[j].v;
        density += computeDensity(p_pos,p_v,q_pos,q_v, params);
    }
    fluid_particles[i].density = density;
    fluid_particles[i].pressure = computePressure(density, params);
}

__device__
double3 computeBoundaryAcceleration(double3 p_pos, double3 k_pos, double3 k_n,
                                    double h, double speed_sound)
{
    double3 p_a;
    double bGamma = boundaryGamma(p_pos,k_pos,k_n,h,speed_sound);
    p_a.x = bGamma * k_n.x;
    p_a.y = bGamma * k_n.y;
    p_a.z = bGamma * k_n.z;

    return p_a;
}

__device__
double3 computeAcceleration(double3 p_pos, double3 p_v, double p_density,
                            double p_pressure, double3 q_pos, double3 q_v,
                            double q_density, double q_pressure, const param *const params)
{
    double3 a;
    double accel;
    double h = params->smoothing_radius;
    double alpha = params->alpha;
    double speed_sound = params->speed_sound;
    double mass_particle = params->mass_particle;
    double surface_tension = params->surface_tension;

    // Pressure force
    accel = (p_pressure/(p_density*p_density) + q_pressure/(q_density*q_density))
            * mass_particle * del_W(p_pos,q_pos,h);
    a.x = -accel * (p_pos.x - q_pos.x);
    a.y = -accel * (p_pos.y - q_pos.y);
    a.z = -accel * (p_pos.z - q_pos.z);

    // Viscosity force
    double VdotR = (p_v.x-q_v.x)*(p_pos.x-q_pos.x)
                 + (p_v.y-q_v.y)*(p_pos.y-q_pos.y)
                 + (p_v.z-q_v.z)*(p_pos.z-q_pos.z);
    if(VdotR < 0.0)
    {
        double nu = 2.0 * alpha * h * speed_sound / (p_density + q_density);
        double r2 = (p_pos.x-q_pos.x)*(p_pos.x-q_pos.x)
                  + (p_pos.y-q_pos.y)*(p_pos.y-q_pos.y)
                  + (p_pos.z-q_pos.z)*(p_pos.z-q_pos.z);
        double eps = h/10.0;
        double stress = nu * VdotR / (r2 + eps*h*h);
        accel = mass_particle * stress * del_W(p_pos, q_pos, h);
        a.x += accel * (p_pos.x - q_pos.x);
        a.y += accel * (p_pos.y - q_pos.y);
        a.z += accel * (p_pos.z - q_pos.z);
    }

    //Surface tension
    // BT 07 http://cg.informatik.uni-freiburg.de/publications/2011_GRAPP_airBubbles.pdf
    accel = surface_tension * W(p_pos,q_pos,h);
    a.x += accel * (p_pos.x - q_pos.x);
    a.y += accel * (p_pos.y - q_pos.y);
    a.z += accel * (p_pos.z - q_pos.z);

    return a;
}

// Update particle acclerations
__global__
void updateAccelerationsFP(fluid_particle *__restrict__ fluid_particles,
                           const param *__restrict__ params)
{
    int num_fluid_particles = params->number_fluid_particles;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_fluid_particles) return;

        double ax = 0.0;
        double ay = 0.0;
        double az = -9.8;

        double3 p_pos = fluid_particles[i].pos;
        double3 p_v   = fluid_particles[i].v;
        double p_density = fluid_particles[i].density;
        double p_pressure = fluid_particles[i].pressure;

        for(int j=0; j<num_fluid_particles; j++) {
            if (i!=j) {
                double3 q_pos = fluid_particles[j].pos;
                double3 q_v   = fluid_particles[j].v;
                double q_density = fluid_particles[j].density;
                double q_pressure = fluid_particles[j].pressure;
                double3 tmp_a = computeAcceleration(p_pos, p_v, p_density,
                                                    p_pressure, q_pos, q_v,
                                                    q_density, q_pressure, params);

                ax += tmp_a.x;
                ay += tmp_a.y;
                az += tmp_a.z;
            }
        }

        fluid_particles[i].a.x = ax;
        fluid_particles[i].a.y = ay;
        fluid_particles[i].a.z = az;
}

__global__
void updateAccelerationsBP(fluid_particle *__restrict__ fluid_particles,
                           const boundary_particle *__restrict__ boundary_particles,
                           const param *__restrict__ params)
{
    int num_fluid_particles = params->number_fluid_particles;
    int num_boundary_particles = params->number_boundary_particles;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_fluid_particles) return;

    double ax = fluid_particles[i].a.x;
    double ay = fluid_particles[i].a.y;
    double az = fluid_particles[i].a.z;
    double3 p_pos = fluid_particles[i].pos;

    for (int j=0; j<num_boundary_particles; j++) {
      double3 k_pos = boundary_particles[j].pos;
      double3 k_n   = boundary_particles[j].n;
      double3 tmp_a = computeBoundaryAcceleration(p_pos,k_pos,k_n,
          params->smoothing_radius,
          params->speed_sound);
      ax += tmp_a.x;
      ay += tmp_a.y;
      az += tmp_a.z;
    }

    fluid_particles[i].a.x = ax;
    fluid_particles[i].a.y = ay;
    fluid_particles[i].a.z = az;
}

// Update particle positions
// Leap Frog integration with v(t+1) estimated
__global__
void updatePositions(fluid_particle *__restrict__ fluid_particles,
                     const param *__restrict__ params)
{
    double dt = params->time_step;

    int num_fluid_particles = params->number_fluid_particles;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_fluid_particles) return;

    // Velocity at t + dt/2
    double3 v_half = fluid_particles[i].v_half;
    double3 v      = fluid_particles[i].v;
    double3 pos    = fluid_particles[i].pos;
    double3 a      = fluid_particles[i].a;

    v_half.x = v_half.x + dt * a.x;
    v_half.y = v_half.y + dt * a.y;
    v_half.z = v_half.z + dt * a.z;

    // Velocity at t + dt, must estimate for foce calc
    v.x = v_half.x + a.x * (dt / 2.0);
    v.y = v_half.y + a.y * (dt / 2.0);
    v.z = v_half.z + a.z * (dt / 2.0);

    // Position at time t + dt
    pos.x = pos.x + dt * v_half.x;
    pos.y = pos.y + dt * v_half.y;
    pos.z = pos.z + dt * v_half.z;

    fluid_particles[i].v_half = v_half;
    fluid_particles[i].v      = v;
    fluid_particles[i].pos    = pos;
}
// ---- end upstream device code -------------------------------------------------

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv) {
  const int n_fluid = 512;       // 8x8x8 lattice
  const int n_boundary = 128;    // 16x8 plane at z=0, normals +z
  const int nsteps = 2;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  param hp;
  hp.rest_density = 1000.0;
  hp.spacing_particle = 0.05;
  hp.mass_particle = hp.rest_density * hp.spacing_particle * hp.spacing_particle * hp.spacing_particle;
  hp.smoothing_radius = 0.1;
  hp.g = 9.8;
  hp.time_step = 1e-4;
  hp.alpha = 0.02;
  hp.surface_tension = 0.01;
  hp.speed_sound = 10.0;
  hp.number_fluid_particles = n_fluid;
  hp.number_boundary_particles = n_boundary;
  hp.number_particles = n_fluid + n_boundary;
  hp.number_steps = nsteps;
  hp.steps_per_frame = 1;

  fluid_particle *fp = (fluid_particle*)malloc(n_fluid * sizeof(fluid_particle));
  boundary_particle *bp = (boundary_particle*)malloc(n_boundary * sizeof(boundary_particle));

  for (int i = 0; i < n_fluid; ++i) {
    int ix = i % 8, iy = (i / 8) % 8, iz = i / 64;
    fp[i].pos.x = 0.025 + ix * 0.05 + 0.005 * (double)h01(i, 1);
    fp[i].pos.y = 0.025 + iy * 0.05 + 0.005 * (double)h01(i, 2);
    fp[i].pos.z = 0.075 + iz * 0.05 + 0.005 * (double)h01(i, 3);
    fp[i].v.x = 0.01 * (2.0 * (double)h01(i, 4) - 1.0);
    fp[i].v.y = 0.01 * (2.0 * (double)h01(i, 5) - 1.0);
    fp[i].v.z = 0.01 * (2.0 * (double)h01(i, 6) - 1.0);
    fp[i].v_half = fp[i].v;
    fp[i].a.x = fp[i].a.y = fp[i].a.z = 0.0;
    fp[i].density = hp.rest_density;
    fp[i].pressure = 0.0;
  }
  for (int j = 0; j < n_boundary; ++j) {
    int jx = j % 16, jy = j / 16;
    bp[j].pos.x = 0.025 + jx * 0.05;
    bp[j].pos.y = 0.025 + jy * 0.05;
    bp[j].pos.z = 0.0;
    bp[j].n.x = 0.0; bp[j].n.y = 0.0; bp[j].n.z = 1.0;
  }

  fluid_particle *d_fp; boundary_particle *d_bp; param *d_p;
  CK(cudaMalloc(&d_fp, n_fluid * sizeof(fluid_particle)));
  CK(cudaMalloc(&d_bp, n_boundary * sizeof(boundary_particle)));
  CK(cudaMalloc(&d_p, sizeof(param)));
  CK(cudaMemcpy(d_fp, fp, n_fluid * sizeof(fluid_particle), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_bp, bp, n_boundary * sizeof(boundary_particle), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_p, &hp, sizeof(param), cudaMemcpyHostToDevice));

  const int tpb = 128;
  const int blocks = (n_fluid + tpb - 1) / tpb;
  for (int s = 0; s < nsteps; ++s) {
    updatePressures<<<blocks, tpb>>>(d_fp, d_p);
    updateAccelerationsFP<<<blocks, tpb>>>(d_fp, d_p);
    updateAccelerationsBP<<<blocks, tpb>>>(d_fp, d_bp, d_p);
    updatePositions<<<blocks, tpb>>>(d_fp, d_p);
  }
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(fp, d_fp, n_fluid * sizeof(fluid_particle), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < n_fluid; ++i)
    fprintf(f, "%.9g\n%.9g\n%.9g\n%.9g\n%.9g\n%.9g\n",
            fp[i].pos.x, fp[i].pos.y, fp[i].pos.z, fp[i].v.x, fp[i].v.y, fp[i].v.z);
  fclose(f);

  cudaFree(d_fp); cudaFree(d_bp); cudaFree(d_p);
  free(fp); free(bp);
  return 0;
}
