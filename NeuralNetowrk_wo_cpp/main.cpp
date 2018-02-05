#include <iostream>
#include <Eigen/Core>
#include <map>
#include <vector>
#include <math.h>
#include <exception>

#pragma region function

enum FunctionTypes
{
	One,
	Identity,
	Binarystep,
	Sign,
	Sigmoid,
	Tanh,
	Relu,
	Leakyrelu,
	Softplus,
	Softsign,
};
typedef double(*activation_function)(double);
typedef std::map<FunctionTypes, activation_function> act_map;

inline double one(double x) { return 1; }
inline double identity(double x) { return x; }
inline double binarystep(double x) { return x > 0 ? 1 : 0; }
inline double sign(double x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); }
inline double sigmoid(double x) { return 1 / (1 + exp(-x)); }
inline double tanh(double x) { return 2 / (1 + exp(-2 * x)) - 1; }
inline double relu(double x) { return x > 0 ? x : 0; }
inline double leakyrelu(double x) { return x > 0 ? x : 0.01 * x; }
inline double softplus(double x) { return log(1 + exp(x)); }
inline double softsign(double x) { return x / (1 + abs(x)); }

act_map functions = {
	{FunctionTypes::One, one},
	{FunctionTypes::Identity, identity},
	{FunctionTypes::Binarystep, binarystep},
	{FunctionTypes::Sign, sign},
	{FunctionTypes::Sigmoid, sigmoid},
	{FunctionTypes::Tanh, tanh},
	{FunctionTypes::Relu, relu},
	{FunctionTypes::Leakyrelu, leakyrelu},
	{FunctionTypes::Softplus, softplus},
	{FunctionTypes::Softsign, softsign},
};

#pragma endregion

typedef Eigen::Matrix<double, -1, 1> RowVector;
typedef Eigen::Matrix<double, -1, -1> Matrix;

struct Neuron
{
public:
	double act_func(double x) { return _act_func(x); }
	Neuron() {};
	Neuron(FunctionTypes ft)
		: _act_func(functions[ft])
	{

	}
	~Neuron() {}
private:
	int z, n;
	double(*_act_func)(double);
};

typedef Eigen::Matrix<Neuron, -1, 1> Layer; // Layer
typedef double Bias; // Bias, len == layer len
typedef Matrix Weight; // Row == (before layer len), Col == (next layer len)

struct NeuralNetwork
{
public:
	NeuralNetwork() {}
	NeuralNetwork(std::vector<int> *v)
	{
		l = NULL;
		b = NULL;
		w = NULL;
		if (v->size() < 2) // input, output
		{
			std::cout << "There should be at least two layer!!" << '\n';
			getchar();
			exit(1);
		}
		else
		{
			try
			{
				l = new Layer[v->size()];
				b = new Bias[v->size() - 1];
				w = new Weight[v->size() - 1];
				for (int i = 0; i < v->size() - 1; i++)
				{
					l[i].resize((*v)[i]);
					for (int j = 0; j < l[i].size(); j++)
					{
						l[i][j] = Neuron(FunctionTypes::Relu);
					}
				}
				l[v->size() - 1].resize(*(v->end() - 1));
				for (int i = 0; i < *(v->end() - 1); i++)
				{
					l[v->size() - 1][i] = Neuron(FunctionTypes::Sigmoid);
				}
				for (int i = 0; i < v->size() - 1; i++)
				{
					w[i].resize((*v)[i] + 1, (*v)[i + 1]);
				}
			}
			catch (std::exception &e)
			{
				std::cout << "Exception #1! " << e.what();
				getchar();
				exit(1);
			}
			std::cout << "Layers :\n";
			for (int i = 0; i < v->size(); i++)
			{
				std::cout << l[i].size() << ' ';
			}
			std::cout << '\n';
		}
	}
	~NeuralNetwork()
	{
		std::cout << "hello ";
		delete[] l;
		std::cout << "world";
		delete[] w;
		std::cout << '!';
		delete[] b;
		std::cout << '!';
		getchar();
	}

	RowVector dCost(RowVector *realvalue, RowVector *estimatedvalue)
	{
		RowVector dcost;
		dcost.resize(realvalue->rows());
		for (int i = 0; i < dcost.size(); i++)
		{
			dcost[i] = (1 - (*realvalue)[i]) / (1 - (*estimatedvalue)[i]) - (*realvalue)[i] / (*estimatedvalue)[i];
		}
		dcost /= dcost.size();
		return dcost;
	}



private:
	Layer *l;
	Bias *b;
	Weight *w;
};

int main()
{
	std::vector<int> v;
	v.push_back(3);
	v.push_back(4);
	v.push_back(5);
	NeuralNetwork n = NeuralNetwork(&v);
	Eigen::Matrix<int, 3, 2> A;
	A << 1, 2, 3, 4, 5, 6;
	std::cout << A;
	getchar(); return 0;
}