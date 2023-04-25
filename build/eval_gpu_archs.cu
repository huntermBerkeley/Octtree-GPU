
#include <cstdio>
#include <set>
#include <string>
using namespace std;
int main(int argc, char** argv) {
  set<string> archs;
  int nDevices;
  if((cudaGetDeviceCount(&nDevices) == cudaSuccess) && (nDevices > 0)) {
    for(int dev=0;dev<nDevices;++dev) {
      char buff[32];
      cudaDeviceProp prop;
      if(cudaGetDeviceProperties(&prop, dev) != cudaSuccess) continue;
      sprintf(buff, "%d%d", prop.major, prop.minor);
      archs.insert(buff);
    }
  }
  //empty arch defaults to 70 for building on V100
  if(archs.empty()) {
    printf("70");
  } else {
    bool first = true;
    for(const auto& arch : archs) {
      printf(first? "%s" : ";%s", arch.c_str());
      first = false;
    }
  }
  printf("\n");
  return 0;
}
