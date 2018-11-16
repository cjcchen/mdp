#include <string.h>
#include <sstream>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <stack>
#include <queue>
#include <set>
 
#include <map>
#include <vector>
#include <string>
#include <stdlib.h>
 
#define ll long long
#define clr(x) memset(x,0,sizeof(x))
#define _clr(x) memset(x,-1,sizeof(x))
#define fr(i,a,b) for(int i = a; i < b; ++i)
#define frr(i,a,b) for(int i = a; i > b; --i)
#define pb push_back
#define sf scanf

#define pf printf
#define mp make_pair
 
using namespace std;

ll mod = 1000000007;

vector<int> g[100010];
int size[100010],son[100010],fa[100010],top[100010],dep[100010];
int sum[100010],add[100010],mul[100010];
int cnt;

void init(){
	clr(sum); clr(add);clr(mul);
	cnt=1;
}

void build_tree(int t, int f) {
	printf("build tree, t = %d f = %d\n",t,f);
	size[t] = 1;
	son[t] = -1;
	int u = -1;
	for(int i = 0; i < g[t].size();++i) {
		if(g[t][i]!=f){
			build_tree(g[t][i],t);
			if(u==-1||size[g[t][i]]>size[u]) {
				u = g[t][i];
			}
			size[t] += size[g[t][i]];
		}
	}
	son[t]=u;
	printf("t = %d f = %d size %d son = %d\n",t,f,size[t],son[t]);
}

void cut_tree(int t, int f, int ffa) {
	if(t==-1){
		return;
	}
	printf("cut t = %d f = %d ffa = %d, son %d\n",t,f,ffa, son[t]);
	top[t] = ffa;
	dep[t] = cnt++;
	fa[t]=f;
	cut_tree(son[t],t,ffa);
	for(int i = 0;i < g[t].size();++i) {
		if(g[t][i]!=f&&g[t][i]!=son[t]){
			cut_tree(g[t][i],t,g[t][i]);
        }
	}
	printf("cut tree, t = %d f = %d ffa = %d top = %d dep = %d\n",t,f,ffa,top[t],dep[t]);
}

void pushdown(int t, int l, int r) {
	mul[t*2] = (mul[t*2]*mul[t])%mod;
	add[t*2] = (add[t*2]*mul[t] + add[t])%mod;

	mul[t*2+1] = (mul[t*2+1]*mul[t])%mod;
	add[t*2+1] = (add[t*2+1]*mul[t] + add[t])%mod;

	int mid = (l+r)>>1;
	sum[t*2] = (sum[t*2]*mul[t]+add[t]*(mid-l+1))%mod;
	sum[t*2+1] = (sum[t*2+1]*mul[t]+add[t]*(r-mid))%mod;

	add[t] = 0;
	mul[t] = 1;
}

void pushup(int t) {
	sum[t] = (sum[t*2]+sum[t*2+1])%mod;
}

void build(int t, int l, int r) {
    sum[t]=0; add[t]=0; mul[t]=1; 
    if(l==r) {
		return;
	}
	int mid = (l+r)>>1;
	build(t*2,l,mid);
	build(t*2+1,mid+1,r);
	pushup(t);
}

void insert(int t,int ql, int qr, int l, int r, int x, int flag) {
    printf("insert seg tree, t = %d ql = %d qr = %d l = %d r = %d x = %d flag = %d\n",
                        t,ql,qr,l,r,x,flag);
	if(ql<=l&&r<=qr) {
		if(flag==2) {//add
			add[t] = (add[t]+x)%mod;
			sum[t] = (sum[t]+x*(r-l+1))%mod;
		}
		else if(flag==1) {
			//mul
			sum[t] =(sum[t]*x)%mod;
			add[t] =(add[t]*x)%mod;
			mul[t] =(mul[t]*x)%mod;
		}
		else{
		}
        printf("leave t = %d l = %d r = %d sum = %d add = %d mul = %d\n",t,l,r,sum[t],add[t],mul[t]);
		return;
	}
	pushdown(t,l,r);
	int mid = (l+r)>>1;
	if(mid>=ql) {
		insert(t*2,ql,min(qr,mid),l,mid,x,flag);
	}
	if(mid<qr) {
		insert(t*2+1,max(mid+1,ql),qr,mid+1,r,x,flag);
	}
	pushup(t);
    printf("t = %d l = %d r = %d sum = %d add %d mul %d\n",t,l,r,sum[t],add[t],mul[t]);
}

ll query(int t, int ql, int qr, int l, int r) {
	if(ql<=l&&qr>=r) {
		return sum[t];
	}
	pushdown(t,l,r);
	int mid = (l+r)>>1;
	ll ret = 0;
	if(mid>=ql) {
		ret=(ret+query(t,ql,min(qr,mid),l,mid))%mod;
	}
	if(mid<qr) {
		ret=(ret+query(t,max(mid+1,ql),qr,mid+1,r))%mod;
	}
	pushup(t);
	return ret;
}

void insert_tree(int t, int u, int v, int x, int flag) {
    printf("insert t = %d u = %d v = %d x = %x flag = %d\n",t,u,v,x,flag);
	ll ret = 0;
	int t1 = top[u],t2 = top[v];
	while(t1!=t2) {
        printf("u = %d v = %d t1 = %d t2 = %d, dep %d, %d\n",u,v,t1,t2,dep[t1],dep[t2]);
		if(dep[t1]>dep[t2]) {
			swap(t1,t2);
			swap(u,v);
		}
		insert(1,dep[t2],dep[v],1,cnt, x, flag);
		v=fa[v];
		t1 = top[u];
		t2 = top[v];
        printf("after u = %d v = %d t1 = %d t2 = %d, dep %d, %d\n",u,v,t1,t2,dep[t1],dep[t2]);
	}
	if(dep[u]>dep[v])swap(u,v);
    printf("u = %d v = %d, dep u = %d v = %d\n",u,v,dep[u],dep[v]);
	insert(1,dep[u],dep[v],1,cnt, x, flag);
	return ;
}

ll query_tree(int t, int u, int v) {
    printf("query t = %d u = %d v = %d\n",t,u,v);
	ll ret = 0;
	int t1 = top[u],t2 = top[v];
	while(t1!=t2) {
		if(dep[t1]>dep[t2]) {
			swap(t1,t2);
			swap(u,v);
		}
		ret += query(1,dep[t2],dep[v],1,cnt);
		v=fa[v];
		t1 = top[u];
		t2 = top[v];
	}
	if(dep[u]>dep[v])swap(u,v);
	ret += query(1,dep[u],dep[v],1,cnt);
	return ret;
}

int main() {
	int n;
	while(sf("%d",&n)>0) {
		init();
		fr(i,0,n)g[i+1].clear();
		fr(i,1,n) {
			int b;
			sf("%d",&b);
			g[b].push_back(i+1);
			printf("b = %d i = %d\n",b,i+1);
		}
		build_tree(1,0);
		cut_tree(1,0,1);
        build(1,1,cnt);
		int m;
		sf("%d",&m);
		fr(i,0,m) {
			int opt;
			sf("%d",&opt);
			if(opt==1||opt==2) {
				int u,v,x;
				sf("%d%d%d",&u,&v,&x);
				insert_tree(1,u,v,x,opt);
			}
			else if(opt==3) {
				int u,v;
				sf("%d%d",&u,&v);
				insert_tree(1,u,v,0,opt);
			}
			else {
				int u,v;
				sf("%d%d",&u,&v);
				ll ans = query_tree(1,u,v);
				printf("%lld\n",ans);
			}
		}
	}
}

