#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Point {
public:
  Point(double x, double y) {
    this->x = x;
    this->y = y;
  }

  double getX() const { return x; }
  double getY() const { return y; }

private:
  double x;
  double y;
};

class KDNode {
public:
  KDNode(Point point, KDNode* left = nullptr, KDNode* right = nullptr) {
    this->point = point;
    this->left = left;
    this->right = right;
  }

  Point getPoint() const { return point; }
  KDNode* getLeft() const { return left; }
  KDNode* getRight() const { return right; }

private:
  Point point;
  KDNode* left;
  KDNode* right;
};

class KDTree {
public:
  KDTree(vector<Point> points) {
    root = buildTree(points, 0);
  }

  KDNode* getRoot() const { return root; }

private:
  KDNode* root;

  KDNode* buildTree(vector<Point>& points, int depth) {
    if (points.empty()) {
      return nullptr;
    }

    int axis = depth % 2;

    sort(points.begin(), points.end(), [axis](const Point& p1, const Point& p2) {
      return p1.get(axis) < p2.get(axis);
    });

    int median = points.size() / 2;
    KDNode* node = new KDNode(points[median]);

    vector<Point> left(points.begin(), points.begin() + median);
    node->setLeft(buildTree(left, depth + 1));

    vector<Point> right(points.begin() + median + 1, points.end());
    node->setRight(buildTree(right, depth + 1));

    return node;
  }
};
