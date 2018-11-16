#include <stdio.h>

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

int fa[400010],dis[400010];
int size[400010];
int node[400010][30];
char s[200005];
int dep[400010];

int cur,cnt,root;

void init(){
    clr(fa);
    clr(node);
    clr(size);
    clr(dis);
    cur = 1;
    cnt = 2;
    root = 1;
}

int new_node(int cur){
    dis[cnt]=dis[cur]+1;
    return cnt++;
}

void insert(int w) {
    int p = cur; cur=new_node(cur);
    size[cur]=1;
    for(; p&&!node[p][w]; p = fa[p]) node[p][w]=cur;
    if(!p){
        fa[cur]=root;
    }
    else {
        int q = node[p][w];
        if(dis[q]==dis[p]+1) {
            fa[cur]=q;
        }
        else {
            int n_n=new_node(p);
            memcpy(node[n_n], node[q], sizeof(node[q]));
            fa[n_n]=fa[q];
            fa[cur]=n_n;
            fa[q]=n_n;
            for(; p&&node[p][w]==q; p =fa[p]) node[p][w]=n_n;
        }
    }
}

void count() {
    clr(dep);
    for(int i = 2; i<cnt;++i) {
        dep[fa[i]]++;
    }
    queue<int> q;
    for(int i = 1; i < cnt; ++i) {
        if(dep[i]==0) {
            q.push(i);
        }
    }
    while(!q.empty()) {
        int t = q.front();q.pop();
        dep[fa[t]]--;
        size[fa[t]]+=size[t];
        if(!dep[fa[t]]){
            q.push(fa[t]);
        }
    }
}

int main() {

    int a,b;
    while(scanf("%s%d%d",&s,&a,&b)>0) {
        init();
        int n = strlen(s);
        for(int i = 0; i < n;++i) {
            insert(s[i]-'A');
        }
        
        count();

        ll ans=0;        
        for(int i = 2; i <cnt;++i) {
            if(size[i]>=a&&size[i]<=b) {
                ans+=dis[i]-dis[fa[i]];
            }
        }
        printf("%lld\n",ans);
    }
}

