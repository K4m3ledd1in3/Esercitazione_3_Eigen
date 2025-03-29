#include <iostream>
#include "Eigen/Eigen"
using namespace std;
using namespace Eigen;
using uint = unsigned int;

void Solve_System_LU(const MatrixXd& L, const MatrixXd& U,const PartialPivLU<MatrixXd>& LU, const MatrixXd& b, MatrixXd& out){
	MatrixXd y = L.inverse()*LU.permutationP()*b;
    out = U.inverse()*y;
}
void Factor_LU(const PartialPivLU<MatrixXd>& LU, MatrixXd& L, MatrixXd& U,const MatrixXd& A){
    U = LU.matrixLU().triangularView<Upper>();     
	L = MatrixXd::Identity(A.rows(), A.cols());
    L.triangularView<StrictlyLower>() = LU.matrixLU();
    cout << " U : " << endl << U << endl << endl;
    cout << " L: " << endl << L << endl << endl;
    cout << "P^-1*L*U " << endl << LU.permutationP().inverse()*L*U << endl << endl;
}
int main()
{	
    MatrixXd U,L,x;
    MatrixXd A(2,2);
	A << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01,
-9.992887623566787e-01 ;
	cout << "A: " << endl << A<<endl<<endl;
	MatrixXd b(2,1);
	b << -5.169911863249772e-01, 1.672384680188350e-01;
	cout << "b: " << endl << b << endl << endl;
	PartialPivLU<MatrixXd> LU(A);
    Factor_LU(LU,L,U,A);
	Solve_System_LU(L,U,LU,b,x);
	cout << x << endl;
	HouseholderQR<MatrixXd> QR(A);
	//
	A << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01,
-8.324762492991313e-01 ;
	cout << "A: " << endl << A<<endl<<endl;
	b << -6.394645785530173e-04, 4.259549612877223e-04;
	cout << "b: " << endl << b << endl << endl;
	PartialPivLU<MatrixXd> LU1(A);
    Factor_LU(LU1,L,U,A);
	Solve_System_LU(L,U,LU1,b,x);
	cout << x << endl;
	//
	A << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01,
-8.320502947645361e-01 ;
	cout << "A: " << endl << A<<endl<<endl;
	b << -6.400391328043042e-10, 4.266924591433963e-10;
	cout << "b: " << endl << b << endl << endl;
	PartialPivLU<MatrixXd> LU2(A);
    Factor_LU(LU2,L,U,A);
	Solve_System_LU(L,U,LU2,b,x);
	cout << x << endl;



 
	return 0;
}


