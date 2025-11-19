#include <iostream>
#include<string>
using namespace std;


class HuffmanNode {
public:
    char data;           
    int freq;           
    HuffmanNode* left;  
    HuffmanNode* right; 

    // 构造
    HuffmanNode(char d, int f) : data(d), freq(f), left(nullptr), right(nullptr) {}
};

// 最小堆类(获得最小值)
class MinHeap {
private:
    HuffmanNode** heap;  // 堆数组
    int capacity;       // 堆容量
    int size;           // 当前堆大小

    // 上浮调整
    void Up(int index) {
        while (index > 0) {
            int parent = (index - 1) / 2;//父节点
            if (heap[parent]->freq <= heap[index]->freq) {
                break;
            }
            // 交换节点
            HuffmanNode* temp = heap[parent];
            heap[parent] = heap[index];
            heap[index] = temp;

            index = parent;
        }
    }
    // 下沉调整
    void Down(int index) {
        int smallest = index;
        int left = 2 * index + 1;
        int right = 2 * index + 2;

        if (left < size && heap[left]->freq < heap[smallest]->freq)
            smallest = left;
        if (right < size && heap[right]->freq < heap[smallest]->freq)
            smallest = right;

        if (smallest != index) {
            // 交换节点
            HuffmanNode* temp = heap[index];
            heap[index] = heap[smallest];
            heap[smallest] = temp;

            Down(smallest);
        }
    }
public:
    // 构造函数
    MinHeap(int cap) : capacity(cap), size(0) {
        heap = new HuffmanNode * [capacity];
    }
    // 析构函数
    ~MinHeap() {
        delete[] heap;
    }
    // 插入节点
    void insert(HuffmanNode* node) {
        if (size == capacity) {
            cout << "满了" << endl;
            return;
        }
        heap[size] = node;
        Up(size);
        size++;
    }
    // 提取最小节点
    HuffmanNode* extractMin() {
        if (size == 0) return nullptr;
        HuffmanNode* minNode = heap[0];
        heap[0] = heap[size - 1];
        size--;
        Down(0);
        return minNode;
    }
    // 获取堆大小
    int getSize() const {
        return size;
    }
};

// Huffman编码类
class HuffmanCoding {
private:
    HuffmanNode* root;          // 树根节点
    string codes[26];           // 储存编码

    // 构建编码表
    void buildCodes(HuffmanNode* node, string code) {
        if (node == nullptr) return;

        // 如果是叶子节点，存储编码
        if (node->left == nullptr && node->right == nullptr) {
            // 确保是大写字母
            if (node->data >= 'A' && node->data <= 'Z') {
                codes[node->data - 'A'] = code;
            }
            return;
        }
        //左子树给0，右子树给1.
        buildCodes(node->left, code + "0");
        buildCodes(node->right, code + "1");
    }

    // 递归删除Huffman树
    void deleteTree(HuffmanNode* node) {
        if (node == nullptr) return;

        deleteTree(node->left);
        deleteTree(node->right);
        delete node;
    }

public:
    // 构造函数
    HuffmanCoding() : root(nullptr) {
        // 初始化编码表为空字符串
        for (int i = 0; i < 26; i++) {
            codes[i] = "";
        }
    }

    // 析构函数
    ~HuffmanCoding() {
        deleteTree(root);
    }

    // 构建Huffman树
    void buildTree(const string& text) {
        // 统计大写字母频率
        int freq[26] = { 0 };
        for (char c : text) {
            if (c >= 'A' && c <= 'Z') {
                freq[c - 'A']++;
            }
        }

        // 创建最小堆
        int uniqueChars = 0;
        for (int i = 0; i < 26; i++) {
            if (freq[i] > 0) uniqueChars++;
        }

   
        MinHeap minHeap(uniqueChars);

        // 将所有出现的大写字母作为叶子节点插入堆中
        for (int i = 0; i < 26; i++) {
            if (freq[i] > 0) {
                minHeap.insert(new HuffmanNode('A' + i, freq[i]));
            }
        }

        // 构建Huffman树
        while (minHeap.getSize() > 1) {
            // 提取两个频率最小的节点
            HuffmanNode* left = minHeap.extractMin();
            HuffmanNode* right = minHeap.extractMin();

            // 创建新节点，频率为两个子节点频率之和
            // 内部节点用'\0'表示
            HuffmanNode* newNode = new HuffmanNode('\0', left->freq + right->freq);
            newNode->left = left;
            newNode->right = right;

            // 将新节点插入堆中
            minHeap.insert(newNode);
        }

        // 最后剩下的节点就是根节点
        root = minHeap.extractMin();

        // 构建编码表
        buildCodes(root, "");
    }

    // 打印所有大写字母的Huffman编码
    void printAllCodes() {
        cout << "Huffman编码表：" << endl;
        bool hasCodes = false;
        for (int i = 0; i < 26; i++) {
            if (!codes[i].empty()) {
                cout << "字母 '" << (char)('A' + i) << "' 的编码: " << codes[i] << endl;
                hasCodes = true;
            }
        }
        if (!hasCodes) {
            cout << "没有生成任何编码！" << endl;
        }
    }

    // 获取特定字母的编码
    string getCode(char c) {
        if (c >= 'A' && c <= 'Z') {
            return codes[c - 'A'];
        }
        return ""; // 非大写字母返回空字符串
    }
    // 编码函数：将大写字母文本转换为Huffman编码
    string encode(const string& text) {
        string encodedText = "";

        for (char c : text) {
            // 只处理大写字母
            if (c >= 'A' && c <= 'Z') {
                encodedText += codes[c - 'A'];
            }
        }
        return encodedText;
    }
};
// 主函数：演示Huffman编码的使用
int main() {
    HuffmanCoding huffman;
    string text;
    cout << "请输入一段大写字母文本: ";
    getline(cin, text);
    if (text.empty()) {
        cout << "输入文本不能为空！" << endl;
        return 1;
    }
    // 构建Huffman树
    huffman.buildTree(text);
    // 显示编码表
    cout << endl;
    huffman.printAllCodes();
    return 0;
}