// ConsoleApplication3.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <tuple>
#include <vector>
#include "utils.hpp"
#include "mpi.h"


using namespace std;
using namespace utils;

int main(int argc, char* argv[])
{


    if (argc == 3) {
        generateMatrix(argv[1], atoi(argv[2]));
    } 
    else if (argc == 4) {
        generateInitX(argv[1], atoi(argv[2]), atof(argv[3]));
    }
    else if (argc != 5) {
        cout << "Укажите все аргументы" << endl;
        return (1);
    }
    else {
        double wtime;
        int n, nBar;
        int* countsALocal, * countsBLocal;
        int* displacementsALocal, * displacementsBLocal; 
        double* aLocal, *bLocal;
        double* a = nullptr, *b = nullptr;
        double* xs = nullptr;

        string matrixFile(argv[1]);
        string xInputFile(argv[2]);
        double e(atof(argv[3]));
        string xOutputFile(argv[4]);


        MPI_Init(&argc, &argv);
        int rank, numprocs;
        MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (numprocs <= 1) {
            throw "Необходимо больше одного процесса";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (rank == 0) {
            tie(a,b, xs, n) = load(matrixFile, xInputFile);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        wtime = MPI_Wtime();

        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            xs = new double[n];
        }
        MPI_Bcast(xs, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        nBar = n / numprocs;
        int residue = n % numprocs;
        countsALocal = new int[numprocs];
        displacementsALocal = new int[numprocs];
        for (int p = 0; p < numprocs; p++) {
            countsALocal[p] = nBar *n;
            displacementsALocal[p] = p * nBar*n;
            if (p<residue) {
                countsALocal[p] += n;
                displacementsALocal[p] += (p+1) * n;
            }
        }
        aLocal = new double[countsALocal[rank]];
        countsBLocal = new int[numprocs];
        displacementsBLocal = new int[numprocs];
        for (int p = 0; p < numprocs; p++) {
            countsBLocal[p] = nBar;
            displacementsBLocal[p] = p * nBar;
            if (p < residue) {
                countsBLocal[p] += 1;
                displacementsBLocal[p] += p+1;
            }
        }
        bLocal = new double[countsBLocal[rank]];
        MPI_Scatterv(a, countsALocal, displacementsALocal, MPI_DOUBLE,
            aLocal, countsALocal[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Scatterv(b, countsBLocal, displacementsBLocal, MPI_DOUBLE,
            bLocal, countsBLocal[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        jacobi(n, countsBLocal[rank], aLocal,bLocal, xs, e, rank, numprocs, countsBLocal, displacementsBLocal);

        double* result = new double[n];
        MPI_Gatherv(xs, countsBLocal[rank], MPI_DOUBLE, result,countsBLocal,displacementsBLocal, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        MPI_Barrier(MPI_COMM_WORLD);
        wtime = MPI_Wtime() - wtime;
        if (rank == 0) {
            char buf[1000];
            int res = -1;
            res = snprintf(buf, sizeof(buf), "Количество процессов = %d, Порядок = %d, Время %f\n", numprocs, n, wtime);
            string str = "Ошибка";
            if (res >= 0 && res < sizeof(buf))
                str = buf;
            cout << str << endl;
            writeLogTime(str);
            printResultInFile(xOutputFile, result, n);

        }
        MPI_Finalize();
    }
    return 0;
}

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
