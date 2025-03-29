#include <iostream>
#include "Eigen/Eigen"
#include <limits>
#include <iomanip>

using namespace std;
using namespace Eigen;
using uint = unsigned int;
const int _prec = 16;

void Solve_System_LU(const MatrixXd& L, const MatrixXd& U,const PartialPivLU<MatrixXd>& LU, const MatrixXd& b, MatrixXd& out);

void Factor_LU(const PartialPivLU<MatrixXd>& LU, MatrixXd& L, MatrixXd& U,const MatrixXd& A);

void Factor_QR(MatrixXd& Q, MatrixXd& R, const MatrixXd& A);

void Solve_System_QR(const MatrixXd& Q, const MatrixXd& R, const MatrixXd& b, MatrixXd& xQR);

void Error(const MatrixXd& X0, const MatrixXd& X, const string& method);

int main(void)
{   
	MatrixXd U,L,x, xQR, Q, R;
	MatrixXd X(2,1);  	
	X << -1,-1;   
	MatrixXd A(2,2);
       std::cout << std::setprecision(_prec) << std::scientific; 
	
	cout << endl << "(1)" << endl;
	A << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01, -9.992887623566787e-01 ;
	MatrixXd b(2,1);
	b << -5.169911863249772e-01, 1.672384680188350e-01;
	PartialPivLU<MatrixXd> LU(A);
        Factor_LU(LU,L,U,A);
	Solve_System_LU(L,U,LU,b,x);
	cout << "xLU: " <<endl<< x << endl;
	Error(X,x,"PA=LU");

	Factor_QR(Q,R,A);
	Solve_System_QR(Q,R,b,xQR);
	Error(X,xQR,"QR");
	
	
	cout << endl << "(2)" << endl;
	A << 5.547001962252291e-01,-5.540607316466765e-01, 8.320502943378437e-01,-8.324762492991313e-01 ;
	b << -6.394645785530173e-04, 4.259549612877223e-04;
	
	PartialPivLU<MatrixXd> LU1(A);
        Factor_LU(LU1,L,U,A);
	Solve_System_LU(L,U,LU1,b,x);
	
	cout << "xLU: " <<endl<< x << endl;
	Error(X,x,"PA=LU");
	
	Factor_QR(Q,R,A);
	Solve_System_QR(Q,R,b,xQR);
	Error(X,xQR,"QR");

	cout << endl << "(3)" << endl;
	A << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01,-8.320502947645361e-01 ;
	b << -6.400391328043042e-10, 4.266924591433963e-10;
	
	PartialPivLU<MatrixXd> LU2(A);
        Factor_LU(LU2,L,U,A);
	Solve_System_LU(L,U,LU2,b,x);
	
	cout << "xLU: " <<endl<< x << endl;
	Error(X,x,"PA=LU");
	
	Factor_QR(Q,R,A);
	Solve_System_QR(Q,R,b,xQR);
	Error(X,xQR,"PA=LU");
	cout << endl;
	
	return 0;
}

void Solve_System_LU(const MatrixXd& L, const MatrixXd& U,const PartialPivLU<MatrixXd>& LU, const MatrixXd& b, MatrixXd& out){
	MatrixXd y = L.inverse()*LU.permutationP()*b;
         out = U.inverse()*y;
}

void Factor_LU(const PartialPivLU<MatrixXd>& LU, MatrixXd& L, MatrixXd& U,const MatrixXd& A){
    U = LU.matrixLU().triangularView<Upper>();     
    L = MatrixXd::Identity(A.rows(), A.cols());
    L.triangularView<StrictlyLower>() = LU.matrixLU();
}

void Factor_QR(MatrixXd& Q, MatrixXd& R, const MatrixXd& A){
	HouseholderQR<MatrixXd> QR(A);
	Q = QR.householderQ();
	R = QR.matrixQR().triangularView<Upper>();

}

void Solve_System_QR(const MatrixXd& Q, const MatrixXd& R, const MatrixXd& b, MatrixXd& xQR){
	MatrixXd y = Q.transpose()*b;
	xQR = R.inverse()*y;
	cout << "xQR:" << endl<<  xQR<<endl;
}
void Error(const MatrixXd& X0, const MatrixXd& X, const string& method){
	cout << "Errore relativo metodo " << method << ": "<<(X-X0).norm()/(X0.norm()) << endl;  
}
