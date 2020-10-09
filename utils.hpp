#include <tuple>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include "mpi.h"

using namespace std;

namespace utils {

	void generateInitX(const string& xsFilename, int rows, double initNum) {
		ofstream xsFile(xsFilename, ios_base::trunc);
		xsFile << rows << endl;
		for (int i = 0; i < rows; i++) {
			xsFile << initNum << endl;
		}
		xsFile.close();
	}

	void generateMatrix(const string& matrixFilename, int rows) {
		vector<vector<double>> matrix(rows, vector<double>((int)(rows + 1), 0));

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < rows + 1; j++) {
				matrix[i][j] = rand();
				matrix[i][i] += matrix[i][j];
			}
			matrix[i][i] = matrix[i][i] + 1;
		}

		ofstream matrixFile(matrixFilename, std::ios_base::trunc);
		matrixFile << rows << " " << rows + 1 << endl;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < (int)(rows + 1); j++) {
				matrixFile << matrix[i][j] << " ";
			}
			matrixFile << endl;
		}
		matrixFile.close();
	}

	tuple<double*, double*, double*, int> load(const string& matrixFilename, const string& xsFilename) {
		int rows, columns, m;
		ifstream matrixFile(matrixFilename);
		ifstream xsFile(xsFilename);
		if (!matrixFile || !xsFile) {
			throw std::runtime_error("Не удалось открыть файл");
		}
		matrixFile >> rows >> columns;
		if (rows < 1 || columns < 2) {
			throw std::runtime_error("Некорректный размер матрицы.");
		}
		xsFile >> m;
		if (rows != m) {
			throw std::runtime_error("Некорректное число х.");
		}
		double* a = new double[(long)rows * rows];
		double* b = new double[rows];
		double* xs = new double[rows];

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < rows; j++) {
				matrixFile >> a[i * rows + j];
			}
			matrixFile >> b[i];
			xsFile >> xs[i];
		}
		matrixFile.close();
		xsFile.close();
		return make_tuple(a, b, xs, rows);
	}

	void printVector(char* title, double xLocal[], int n)
	{
		printf("%s\n", title);
		for (int i = 0; i < n; i++)
			printf("%4.1f ", xLocal[i]);
		printf("\n");
	}

	void printResultInFile(const string& outFile, double* matrix, int n) {
		ofstream fout(outFile);
		fout << n << endl;
		for (int i = 0; i < n; i++) {
			fout << matrix[i] << endl;
		}
		fout.close();
	}

	void writeLogTime(string& str) {
		ofstream filelog("log.txt", ios::out | ios::app);
		filelog << str;
		filelog.close();
	}

	double distance(double* x, double* y, int n)
	{
		double sum = 0.0;
		for (int i = 0; i < n; i++) {
			sum = sum + (x[i] - y[i]) * (x[i] - y[i]);

		}
		return sqrt(sum);
	}

	void jacobi(int n, int countRows, double* aLocal, double* bLocal, double* xs, double e, int rank, int numprocs, int* countBLocal, int* displacementsBLocal) {
		double* xOld = new double[n];
		double* xNew = new double[n];
		MPI_Allgatherv(xs, countRows, MPI_DOUBLE, xNew, countBLocal, displacementsBLocal, MPI_DOUBLE, MPI_COMM_WORLD);

		do {
			double* temp = xOld;
			xOld = xNew;
			xNew = temp;
			for (int iLocal = 0; iLocal < countRows; iLocal++) {
				xs[iLocal] = bLocal[iLocal];
				for (int j = 0; j < n; j++) {
					if (j != iLocal + displacementsBLocal[rank]) {
						xs[iLocal] -= aLocal[iLocal * n + j] * xOld[j];
					}
				}
				xs[iLocal] = xs[iLocal] / aLocal[iLocal * n + iLocal + displacementsBLocal[rank]];
			}
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Allgatherv(xs, countRows, MPI_DOUBLE, xNew, countBLocal, displacementsBLocal, MPI_DOUBLE, MPI_COMM_WORLD);

		} while (distance(xNew, xOld, n) >= e);
	}
}