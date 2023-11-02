#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <float.h>
#include <cmath>
using namespace std;

//Helper function to return a random score for each node to use during initialization
class Node;
class Problem;
class Validator;
class Classifier;

vector<vector<double>*> PullData(string filename);

//This class is the Nearest Neighbor algorithm. It takes in a dataset as input, trains itself by just storing the data, and tests by 
//taking a combination and a line of data, and returning the calculated class label
class Classifier {
  public:
    //Takes in a dataset and stores it
    void Train(vector<vector<double>*> temp) {
      data = temp;
    }

    //This function is overloaded to have two versions of the function. The first one takes in a combination and a line of data as input
    //It then compares each entry of the line (except the first) to the stored dataset. It measures the distance from each entry, finds the nearest entry
    //And returns that entry's class label. This version of the function only compares the features detailed in the combination. 
    int Test(string combination, vector<double> input) {
      vector<double> distances;
      double temp = 0;
      int tempIndex = 0;
      for (int i = 0; i < data.size(); i++) {
        for (int j = 1; j < input.size(); j++) {
          if (combination[combination.size() - j] != '0') {
            temp += pow((input[j]-data[i][0][j]), 2);
          }
        }
        distances.push_back(sqrt(temp));
        temp = 0;
      }
      temp = DBL_MAX;
      for (int i = 0; i < distances.size(); i++) {
        if (distances[i] < temp) {
          temp = distances[i];
          tempIndex = i;
        }
      }
      return data[tempIndex][0][0];
    }

    //This version of the function is the same but it compares all features. This was only used for testing. It serves no purpose in the current state of the program
    int Test(vector<double> input) {
      vector<double> distances;
      double temp = 0;
      int tempIndex = 0;
      for (int i = 0; i < data.size(); i++) {
        for (int j = 1; j < input.size(); j++) {
          temp += pow((input[j]-data[i][0][j]), 2);
        }
        distances.push_back(sqrt(temp));
        temp = 0;
      }
      temp = DBL_MAX;
      for (int i = 0; i < distances.size(); i++) {
        if (distances[i] < temp) {
          temp = distances[i];
          tempIndex = i;
        }
      }
      return data[tempIndex][0][0];
    }

  private:
    vector<vector<double>*> data;
};

//This class has only one function and nothing else: Eval(). This function takes in a classifier, a dataset, and a combination as input. 
class Validator{
  public: 

    //This function iterates through each list in the database. It removes that list from the base, inputs the folded data into the validator's Train() function,
    //and uses the folded out data as input for the Validator's test() function. It also gives the combination as input, and only tests using the specified features. 
    //It takes the returned label, compares it to the input data's label, and checks if it is correct. If so, it adds to a count. This count is then returned 
    //to be used as an accuracy value.
    double Eval(Classifier temp, vector <vector<double>*> data, string comb) {
      int correct = 0;
      int size = 0;
      vector <vector<double>*> db = data;
      vector <double> v;
      vector <double>* tv;
      for (int i = 0; i < data.size(); i++) {
        v = db.back()[0];
        db.pop_back();
        temp.Train(db);
        if (v[0] == temp.Test(comb, v)) {
          correct++;
        }
        tv = new vector<double>;
        for (int i = 0; i < v.size(); i++) {
          tv->push_back(v[i]);
        }
        db.insert(db.begin(), tv);
      }
      for (int i = 0; i < db.size(); i++) {
        db.pop_back();
      }
      tv = nullptr;
      return correct;
    }
};

//Node class
//Each node has a combination of features represented as a binary string, a score,
//A size variable that keeps track of how many features are being used in each node, 
//A list of parents and a list of children nodes.
class Node {
  public:
    //Initialize the score, size, and combination to be set later through helper functions
    Node() {
      score = 0;
      size = 0;
      combination = "";
    }

    //Set the combination and size through how many features are present in the combination
    void SetComb(string temp) {
      size = 0;
      combination = temp;
      for (int i = 0; i < temp.size(); i++) {
        if (temp[i] == '1') {
          size++;
        }
      }
    }

    //Set the score for the node
    void SetScore(double temp) {
      score = temp;
    }

    //Push a child to the list
    void PushChild(Node* temp) {
      children.push_back(temp);
    }

    //Return the node's score
    double RetScore() {
      return score;
    }

    //Return the node's size
    int RetSize() {
      return size;
    }

    //Return the node's combination
    string RetComb() {
      return combination;
    }

    //Lists of parents and children
    vector <Node*> children;

  private:
    int size;
    double score;
    string combination;
};

//Problem class that stores the search functions
class Problem {
  public:
    //Constructor which keeps track of the root node, the bottommost (leaf) node, and which algorithm is being used
    Problem(Node* n1, bool temp) {
      root = n1;
      algorithm = temp;
    }

    //Returns either search algorithm depending on which search is selected
    Node* Search(int SIZE, vector<vector<double>*> data, int temp) {
      if (algorithm) {
        return ForwardSearch(SIZE, data, temp);
      } 
      else {
        return BackwardElimination(SIZE, data, temp);
      }
    }

  private:

    //Forward search algorithm
    Node* ForwardSearch(int SIZE, vector<vector<double>*> data, int size) {
      //Helper variables
      Node* temp = root;
      Node* temp2 = nullptr;
      Node* temp3 = nullptr;
      Node* best = nullptr;
      double score = 0;
      double bestScore = 0;
      double lastScore = 0;
      string s;
      Classifier alg;
      Validator test;
      temp->SetScore(test.Eval(alg, data, temp->RetComb())/size);
      for (int i = 0; i < SIZE; i++) {
        s = temp->RetComb();
        s.replace(i, 1, "1");
        if (s != temp->RetComb()) {
          temp3 = new Node();
          temp3->SetComb(s);
          temp3->SetScore(test.Eval(alg, data, temp3->RetComb())/size);
          temp->PushChild(temp3);
        }
      }

      //Initial couts
      cout << "\nUsing no features and \"random\" evaluation, I get an accuracy of " << temp->RetScore()*100 << "%\n";
      cout << "\nFeature sets are represented as binary strings. Each bit represents one feature, \nwith the rightmost bit being feature 1, and the leftmost being the final feature\n";
      cout << "\nBeginning search.\n\n";

      //Main search loop. Runs until the current node has no children
      while (temp->children.size()) {
        for (int i = 0; i < temp->children.size(); i++) {
        cout << "Using feature(s) {" << temp->children[i]->RetComb() << "} accuracy is " << temp->children[i]->RetScore()*100 << "%\n";
          if (temp->children[i]->RetScore() > score) {
            score = temp->children[i]->RetScore();
            temp2 = temp->children[i];
            if (score > bestScore) {
              bestScore = score;
              best = temp2;
            }
          }
        }

        //Shows which feature set was selected, and displays a warning if the accuracy goes down between iterations
        cout << "\n" << "Feature set {" << temp2->RetComb() << "} was best, accuracy is " << temp2->RetScore()*100 << "%\n\n";
        if (temp2->RetScore() < lastScore) {
          cout << "(Warning, Accuracy has decreased!)\n\n";
        }
        temp = temp2;
        lastScore = score;
        score = 0;
        if (temp->RetSize() < SIZE) {
          s = "";
          for (int i = 0; i < SIZE; i++) {
            s = temp->RetComb();
            s.replace(i, 1, "1");
            if (s != temp->RetComb()) {
              temp3 = new Node();
              temp3->SetComb(s);
              temp3->SetScore(test.Eval(alg, data, temp3->RetComb())/size);
              temp->PushChild(temp3);
            }
          }  
        }
      }

      //Checks the last node's score since it isn't properly checked in the loop
      if (temp2 != nullptr) {
        if (temp2->RetScore() > bestScore) {
          best = temp2;
        }
      }
      //Checks the first node's score since it isn't properly checked in the loop
      if (root->RetScore() > best->RetScore()) {
        best = root;
      }
      return best;
    }

    //Backward elimination algorithm
    Node* BackwardElimination(int SIZE, vector<vector<double>*> data, int size) {
      //Helper variables
      Node* temp = root;
      Node* temp2 = nullptr;
      Node* temp3 = nullptr;
      Node* best = nullptr;
      double score = 0;
      double bestScore = 0;
      double lastScore = 0;
      string s;
      Classifier alg;
      Validator test;
      temp->SetScore(test.Eval(alg, data, temp->RetComb())/size);
      
      for (int i = 0; i < SIZE; i++) {
        s = temp->RetComb();
        s.replace(i, 1, "0");
        if (s != temp->RetComb()) {
          temp3 = new Node();
          temp3->SetComb(s);
          temp3->SetScore(test.Eval(alg, data, temp3->RetComb())/size);
          temp->PushChild(temp3);
        }
      }
      
      //Initial couts
      cout << "\nUsing all features, I get an accuracy of " << temp->RetScore()*100 << "%\n";
      cout << "\nFeature sets are represented as binary strings. \nEach bit represents one feature, \nwith the rightmost bit being feature 1, and the leftmost being the final feature\n";
      cout << "\nBeginning search.\n\n";
  
      //Main search loop. Runs until the current node has no children
      while (temp->children.size()) {
        //Iterate through the node's children and find the best score among them
        for (int i = 0; i < temp->children.size(); i++) {
        cout << "Using feature(s) {" << temp->children[i]->RetComb() << "} accuracy is " << temp->children[i]->RetScore()*100 << "%\n";
          if (temp->children[i]->RetScore() > score) {
            score = temp->children[i]->RetScore();
            temp2 = temp->children[i];
            if (score > bestScore) {
              bestScore = score;
              best = temp2;
            }
          }
        }
        
        //Shows which feature set was selected, and displays a warning if the accuracy goes down between iterations
        cout << "\n" << "Feature set {" << temp2->RetComb() << "} was best, accuracy is " << temp2->RetScore()*100 << "%\n\n";
        if (temp2->RetScore() < lastScore) {
          cout << "(Warning, Accuracy has decreased!)\n\n";
        }
        
        temp = temp2;
        lastScore = score;
        score = 0;
        
        if (temp->RetSize() < SIZE) {
          s = "";
          for (int i = 0; i < SIZE; i++) {
            s = temp->RetComb();
            s.replace(i, 1, "0");
            if (s != temp->RetComb()) {
              temp3 = new Node();
              temp3->SetComb(s);
              temp3->SetScore(test.Eval(alg, data, temp3->RetComb())/size);
              temp->PushChild(temp3);
            }
          }  
        }
      }
      
      //Checks the last node's score since it isn't properly checked in the loop
      if (temp2 != nullptr) {
        if (temp2->RetScore() > bestScore) {
          best = temp2;
        }
      }
      
      //Checks the first node's score since it isn't properly checked in the loop
      if (root->RetScore() > best->RetScore()) {
        best = root;
      }
      return best;
    }

    bool algorithm;
    Node* root;
};

//Helper function that pulls data from a specified file, formats it as a vector of vector <double> pointers. This format can be used by my algorithms.
vector<vector<double>*> PullData(string filename) {
  fstream file;
  string word, temp;
  vector <double>* tempdata;
  vector <vector<double>*> data;
  int number = 0;
  file.open(filename.c_str());
  while (getline(file, temp)) {
    tempdata = new vector <double>;
    istringstream line(temp);
    while (line >> word) {
      tempdata->push_back(stod(word));
    }
    data.push_back(tempdata);
  }
  tempdata = nullptr;
  return data;
}

//Helper function that takes a database and its size as input and returns a normalized database
vector<vector<double>*> Normalize (vector<vector<double>*> data, int size) {
  vector <double> min_vals;
  vector <double> max_vals;
  vector<vector<double>*> normal_data;
  int SIZE = size; 

  //Vectors that store the max and min values for each feature are initialized
  for (int i = 0; i < SIZE; i++) {
    min_vals.push_back(DBL_MAX);
    max_vals.push_back(0);
  }

  //Max and min values are found
  for (int i = 1; i < data[0][0].size(); i++) {
    for (int j = 0; j < data.size(); j++) {
      if (data[j][0][i] < min_vals[i]) {
        min_vals[i] = data[j][0][i];
      }
      if (data[j][0][i] > max_vals[i]) {
        max_vals[i] = data[j][0][i];
      }
    }
  }

  //The normalized dataset is initialized to the initial data set
  for (int i = 0; i < data.size(); i++) {
    normal_data.push_back(new vector<double>);
    normal_data[i]->push_back(data[i][0][0]);
    for (int j = 1; j < data[0][0].size(); j++) {
      normal_data[i][0].push_back((data[i][0][j]-min_vals[j])/(max_vals[j]-min_vals[j]));
    }
  }
/*
  //The new dataset is then normalized using the max and min values
  for (int i = 0; i < data.size(); i++) {
    for (int j = 1; j < data[0][0].size(); j++) {
      normal_data[i][0][j] = (data[i][0][j]-min_vals[j])/(max_vals[j]-min_vals[j]);
    }
  }*/
  //Normalized dataset is returned
  return normal_data;
}

int main() {
  //Helper variables for driver code
  Node* temp = nullptr;
  Node* root = nullptr;
  int SIZE;
  bool algorithm;
  string input;
  string s = "";
  //Welcome message and menu, takes filename as input
  cout << "Welcome to Bahman Baidar's Feature Selection Algorithm.\n\n" << "Type in the name of the file to test: ";
  cin >> input;
  vector<vector<double>*> data = PullData(input);
  SIZE = (data[0][0].size()-1);

  //This section of code is responsible for normalizing the data. It uses a helper function that returns a normalized set of data from 
  //an input database and its size
  cout << "This dataset has " << SIZE << " features (not including the class attribute), with " << data.size() << " instances";
  cout << "\n\nPlease wait while I normalize the data... ";
  vector<vector<double>*> normal_data = Normalize(data, SIZE);
  cout << "Done!";
  
  //Algorithm selection, uses input validation
  cout << "\n\nType the number of the algorithm you want to run.\n\n\t1: Forward Selection\n\t2: Backward Elimination\n\n";
  do {
    cin >> input;
    if (input != "1" && input != "2") {
      cout << "\nInvalid input, try again: ";
    }
  } while (input != "1" && input != "2");
  if (input == "1") {
    algorithm = 1;
  }
  else {
    algorithm = 0;
  }

  //Depending on the algorithm selected, the initial node is created
  if (algorithm) {
    s.insert(s.begin(), SIZE, '0');
  }
  else {
    s.insert(s.begin(), SIZE, '1');
  }
  root = new Node();
  root->SetComb(s);
  
  //Initializes the problem algorithm, 
  Problem alg(root, algorithm);

  //Finds the best node from the search and is returned to temp
  temp = alg.Search(SIZE, normal_data, normal_data.size());
  
  //Final selected node's combination and score is displayed
  cout << "Finished search!! The best feature subset is {"<< temp->RetComb() << "}, which has an accuracy of " << temp->RetScore()*100 << "%";/*
  ofstream file1;
  ofstream file2;
  ofstream file3;
  ofstream file4;
  file1.open("data1-3.txt");
  file2.open("data1-5.txt");
  file3.open("data2-3.txt");
  file4.open("data2-5.txt");
  for (int i = 0; i < normal_data.size(); i++) {
    if (data[i][0][0] == 1) {
      file1 << normal_data[i][0][8] << endl;
      file2 << normal_data[i][0][6] << endl;
    }
    else {
      file3 << normal_data[i][0][8] << endl;
      file4 << normal_data[i][0][6] << endl;
    }
  }*/
  return 0;
}