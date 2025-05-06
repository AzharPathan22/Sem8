#include <iostream>
#include <queue>
#include <omp.h>
#include <vector>

using namespace std;

// Tree node
class Node {
public:
    int data;
    Node* left;
    Node* right;

    Node(int value) {
        data = value;
        left = NULL;
        right = NULL;
    }
};

// Sample binary tree
Node* createSampleTree() {
    Node* root = new Node(1);
    root->left = new Node(2);
    root->right = new Node(3);
    root->left->left = new Node(4);
    root->left->right = new Node(5);
    root->right->left = new Node(6);
    root->right->right = new Node(7);
    return root;
}

// Parallel BFS (Level Order Traversal)
void parallelBFS(Node* root) {
    if (!root) return;

    queue<Node*> q;
    q.push(root);

    cout << "Parallel BFS Traversal: ";
    while (!q.empty()) {
        int size = q.size();
        vector<Node*> levelNodes;

        // Collect current level nodes
        for (int i = 0; i < size; ++i) {
            Node* node = q.front(); q.pop();
            levelNodes.push_back(node);
        }

        // Parallel print and enqueue next level
        #pragma omp parallel for
        for (int i = 0; i < levelNodes.size(); ++i) {
            #pragma omp critical
            cout << levelNodes[i]->data << " ";

            #pragma omp critical
            {
                if (levelNodes[i]->left) q.push(levelNodes[i]->left);
                if (levelNodes[i]->right) q.push(levelNodes[i]->right);
            }
        }
    }
    cout << endl;
}

// Parallel DFS (Pre-order)
void parallelDFS(Node* root) {
    if (!root) return;

    #pragma omp critical
    cout << root->data << " ";

    #pragma omp parallel sections
    {
        #pragma omp section
        parallelDFS(root->left);

        #pragma omp section
        parallelDFS(root->right);
    }
}

int main() {
    Node* root = createSampleTree();

    double start_bfs = omp_get_wtime();
    parallelBFS(root);
    double end_bfs = omp_get_wtime();
    cout << "Parallel BFS Time: " << (end_bfs - start_bfs) * 1000 << " ms" << endl;

    double start_dfs = omp_get_wtime();
    cout << "Parallel DFS Traversal: ";
    parallelDFS(root);
    cout << endl;
    double end_dfs = omp_get_wtime();
    cout << "Parallel DFS Time: " << (end_dfs - start_dfs) * 1000 << " ms" << endl;

    return 0;
}

