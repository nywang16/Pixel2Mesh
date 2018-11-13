#include <cstdio>
#include <algorithm>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <math.h>
#include <string.h>
using namespace std;
float randomf(){
	return (rand()+0.5)/(RAND_MAX+1.0);
}
static double get_time(){
	timespec tp;
	clock_gettime(CLOCK_MONOTONIC,&tp);
	return tp.tv_sec+tp.tv_nsec*1e-9;
}
void approxmatch_cpu(int b,int n,int m,float * xyz1,float * xyz2,float * match){
	for (int i=0;i<b;i++){
		int factorl=max(n,m)/n;
		int factorr=max(n,m)/m;
		vector<double> saturatedl(n,double(factorl)),saturatedr(m,double(factorr));
		vector<double> weight(n*m);
		for (int j=0;j<n*m;j++)
			match[j]=0;
		for (int j=7;j>=-2;j--){
			//printf("i=%d j=%d\n",i,j);
			double level=-powf(4.0,j);
			if (j==-2)
				level=0;
			for (int k=0;k<n;k++){
				double x1=xyz1[k*3+0];
				double y1=xyz1[k*3+1];
				double z1=xyz1[k*3+2];
				for (int l=0;l<m;l++){
					double x2=xyz2[l*3+0];
					double y2=xyz2[l*3+1];
					double z2=xyz2[l*3+2];
					weight[k*m+l]=expf(level*((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2)))*saturatedr[l];
				}
			}
			vector<double> ss(m,1e-9);
			for (int k=0;k<n;k++){
				double s=1e-9;
				for (int l=0;l<m;l++){
					s+=weight[k*m+l];
				}
				for (int l=0;l<m;l++){
					weight[k*m+l]=weight[k*m+l]/s*saturatedl[k];
				}
				for (int l=0;l<m;l++)
					ss[l]+=weight[k*m+l];
			}
			for (int l=0;l<m;l++){
				double s=ss[l];
				double r=min(saturatedr[l]/s,1.0);
				ss[l]=r;
			}
			vector<double> ss2(m,0);
			for (int k=0;k<n;k++){
				double s=0;
				for (int l=0;l<m;l++){
					weight[k*m+l]*=ss[l];
					s+=weight[k*m+l];
					ss2[l]+=weight[k*m+l];
				}
				saturatedl[k]=max(saturatedl[k]-s,0.0);
			}
			for (int k=0;k<n*m;k++)
				match[k]+=weight[k];
			for (int l=0;l<m;l++){
				saturatedr[l]=max(saturatedr[l]-ss2[l],0.0);
			}
		}
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
	}
}
void matchcost_cpu(int b,int n,int m,float * xyz1,float * xyz2,float * match,float * cost){
	for (int i=0;i<b;i++){
		double s=0;
		for (int j=0;j<n;j++)
			for (int k=0;k<m;k++){
				float x1=xyz1[j*3+0];
				float y1=xyz1[j*3+1];
				float z1=xyz1[j*3+2];
				float x2=xyz2[k*3+0];
				float y2=xyz2[k*3+1];
				float z2=xyz2[k*3+2];
				float d=sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1))*match[j*m+k];
				s+=d;
			}
		cost[0]=s;
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
		cost+=1;
	}
}
void matchcostgrad_cpu(int b,int n,int m,float * xyz1,float * xyz2,float * match,float * grad2){
	for (int i=0;i<b;i++){
		for (int j=0;j<m;j++){
			float sx=0,sy=0,sz=0;
			for (int k=0;k<n;k++){
				float x2=xyz2[j*3+0];
				float y2=xyz2[j*3+1];
				float z2=xyz2[j*3+2];
				float x1=xyz1[k*3+0];
				float y1=xyz1[k*3+1];
				float z1=xyz1[k*3+2];
				float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
				sx+=match[k*m+j]*((x2-x1)/d);
				sy+=match[k*m+j]*((y2-y1)/d);
				sz+=match[k*m+j]*((z2-z1)/d);
			}
			grad2[j*3+0]=sx;
			grad2[j*3+1]=sy;
			grad2[j*3+2]=sz;
		}
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
		grad2+=m*3;
	}
}
void approxmatchLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,float * match);
void matchcostLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * out);
void matchcostgradLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * grad2);
int main()
{
	srand(101);
	int b=32,n=4096,m=n/4;
	float * xyz1=new float[b*n*3];
	float * xyz2=new float[b*m*3];
	float * match=new float[b*n*m];
	float * match_cpu=new float[b*n*m];
	float * cost=new float[b];
	float * cost_cpu=new float[b];
	float * grad=new float[b*m*3];
	float * grad_cpu=new float[b*m*3];
	for (int i=0;i<b*n*3;i++)
		xyz1[i]=randomf();
	for (int i=0;i<b*m*3;i++)
		xyz2[i]=randomf();
	double t0=get_time();
	approxmatch_cpu(2,n,m,xyz1,xyz2,match_cpu);
	printf("approxmatch cpu time %f\n",get_time()-t0);
	/*for (int i=0;i<b;i++){
		for (int j=0;j<n;j++){
			float s=0;
			for (int k=0;k<m;k++){
				float u=match_cpu[i*n*m+j*m+k];
				if (u<0 || u>1){
					printf("bad i=%d j=%d k=%d u=%f\n",i,j,k,u);
				}
				s+=u;
			}
			if (s<0.999 || s>1.001){
				printf("bad i=%d j=%d s=%f\n",i,j,s);
			}
		}
		for (int j=0;j<m;j++){
			float s=0;
			for (int k=0;k<n;k++){
				s+=match_cpu[i*n*m+k*m+j];
			}
			if (s<3.999 || s>4.001){
				printf("bad i=%d j=%d s=%f\n",i,j,s);
			}
		}
	}*/
	/*for (int j=0;j<n;j++){
		for (int k=0;k<m;k++)
			printf("%.3f ",match_cpu[j*m+k]);
		puts("");
	}*/
	matchcost_cpu(2,n,m,xyz1,xyz2,match_cpu,cost_cpu);
	matchcostgrad_cpu(2,n,m,xyz1,xyz2,match_cpu,grad_cpu);

	float * xyz1_g;
	cudaMalloc(&xyz1_g,b*n*3*4);
	float * xyz2_g;
	cudaMalloc(&xyz2_g,b*m*3*4);
	float * match_g;
	cudaMalloc(&match_g,b*n*m*4);
	float * cost_g;
	cudaMalloc(&cost_g,b*n*3*4);
	float * grad_g;
	cudaMalloc(&grad_g,b*m*3*4);

	cudaMemcpy(xyz1_g,xyz1,b*n*3*4,cudaMemcpyHostToDevice);
	cudaMemcpy(xyz2_g,xyz2,b*m*3*4,cudaMemcpyHostToDevice);
	cudaMemset(match_g,0,b*n*m*4);
	cudaMemset(cost_g,0,b*4);
	cudaMemset(grad_g,0,b*m*3*4);
	
	double besttime=0;
	for (int run=0;run<10;run++){
		double t1=get_time();
		approxmatchLauncher(b,n,m,xyz1_g,xyz2_g,match_g);
		matchcostLauncher(b,n,m,xyz1_g,xyz2_g,match_g,cost_g);
		matchcostgradLauncher(b,n,m,xyz1_g,xyz2_g,match_g,grad_g);
		cudaDeviceSynchronize();
		double t=get_time()-t1;
		if (run==0 || t<besttime)
			besttime=t;
		printf("run=%d time=%f\n",run,t);
	}
	printf("besttime=%f\n",besttime);
	memset(match,0,b*n*m*4);
	memset(cost,0,b*4);
	memset(grad,0,b*m*3*4);
	cudaMemcpy(match,match_g,b*n*m*4,cudaMemcpyDeviceToHost);
	cudaMemcpy(cost,cost_g,b*4,cudaMemcpyDeviceToHost);
	cudaMemcpy(grad,grad_g,b*m*3*4,cudaMemcpyDeviceToHost);
	double emax=0;
	bool flag=true;
	for (int i=0;i<2 && flag;i++)
		for (int j=0;j<n && flag;j++){
			for (int k=0;k<m && flag;k++){
				//if (match[i*n*m+k*n+j]>1e-3)
				if (fabs(double(match[i*n*m+k*n+j]-match_cpu[i*n*m+j*m+k]))>1e-2){
					printf("i %d j %d k %d m %f %f\n",i,j,k,match[i*n*m+k*n+j],match_cpu[i*n*m+j*m+k]);
					flag=false;
					break;
				}
				//emax=max(emax,fabs(double(match[i*n*m+k*n+j]-match_cpu[i*n*m+j*m+k])));
				emax+=fabs(double(match[i*n*m+k*n+j]-match_cpu[i*n*m+j*m+k]));
			}
		}
	printf("emax_match=%f\n",emax/2/n/m);
	emax=0;
	for (int i=0;i<2;i++)
		emax+=fabs(double(cost[i]-cost_cpu[i]));
	printf("emax_cost=%f\n",emax/2);
	emax=0;
	for (int i=0;i<2*m*3;i++)
		emax+=fabs(double(grad[i]-grad_cpu[i]));
	//for (int i=0;i<3*m;i++){
		//if (grad[i]!=0)
			//printf("i %d %f %f\n",i,grad[i],grad_cpu[i]);
	//}
	printf("emax_grad=%f\n",emax/(2*m*3));

	cudaFree(xyz1_g);
	cudaFree(xyz2_g);
	cudaFree(match_g);
	cudaFree(cost_g);
	cudaFree(grad_g);

	return 0;
}

