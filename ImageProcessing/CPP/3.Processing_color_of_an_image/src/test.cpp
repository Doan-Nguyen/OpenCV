#include <iostream> 

using namespace std;

void increase(int x){
    x++;
    cout << x << endl;
}

void increase_reference(int &x){
    x++;
    cout << x << endl;
}

void increase_reference2(int& x){
    x++;
    cout << x << endl;
}

int main(){
    int x = 10;

    increase(x);
    cout << x << endl;
    increase_reference(x);
    cout << x << endl;
    increase_reference2(x);
    cout << x << endl;



    return 0;
}