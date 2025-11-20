#include<iostream>
using namespace std;


//定义单链表结点
struct Node {
	int data;//存储数据
	Node* next;//下个结点
	Node(int x=0):data(x),next(nullptr){}
};

//创建有头结点的单链表
Node* CreateList(int t) {
	Node* head = new Node();//头结点
	Node* tail = head;//尾结点
	int x;
	cout << "请输入结点数值"<<":";
	for (int i = 0;i < t;i++) {
		cin >> x;
		tail-> next = new Node(x);
		tail = tail->next;
	}
	return head;//返回头结点
}
//打印链表
void printList(Node* head) {
	Node* p = head->next;
	while (p) {
		cout << p->data << " ";
		p = p->next;
	}
	cout << endl;
}


//找倒数第k个结点
Node* find(Node* head, int k) {
    if (!head->next || k <= 0) return nullptr;

    Node* p = head->next;
    Node* q = head->next;

    // q 先走 k 步
    for (int i = 0; i < k; ++i) {
        if (!q) return nullptr; // 说明链表长度不足 k
        q = q->next;
    }

    // p，q之间差k，p、q 同时前进
    while (q) {//q为空说明，p为倒数第k个结点
        p = p->next;
        q = q->next;
    }

    return p; // p 指向倒数第k个结点
}

int main() {
    int t;
    cout << "请输入结点个数" << ":";
    cin >> t;
    while (!t) {
        cout << "链表不能为空,重新输入";
        cin >>t;
    }
    Node* head = CreateList(t);
    cout << "当前链表为: ";
    printList(head);

    int k;
    cout << "请输入要查找的倒数第k个位置：";
    cin >> k;
    if (k <= 0) {
        cout << "k值非法";
        return 0;
    }

    Node* result = find(head, k);
    if (result)
        cout << "倒数第 " << k << " 个结点的值为: " << result->data << endl;
    else
        cout << "查找失败：链表长度不足 " << k << endl;
     
        
    
    

    return 0;
}
