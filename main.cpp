#include "cv.h"
#include "highgui.h"

using namespace std;
using namespace cv;

int main(int argc, char ** argv)
{
    Mat Mag;
    Mat Phase;

    Mat bild=imread("/home/baiysh/ClionProjects/Furie/audrey.jpg",0);
    imshow("Original",bild);
    Mat ebenen[2];

    int M = getOptimalDFTSize( bild.rows );
    int N = getOptimalDFTSize( bild.cols );

    Mat gepolstert;
    copyMakeBorder(bild, gepolstert, 0, M - bild.rows, 0, N - bild.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat neuEbenen[] = {Mat_<float>(gepolstert), Mat::zeros(gepolstert.size(), CV_32F)};
    Mat komplexBild;
    merge(neuEbenen, 2, komplexBild);
    dft(komplexBild, komplexBild);
    split(komplexBild, neuEbenen);
    neuEbenen[0] = neuEbenen[0](Rect(0, 0, neuEbenen[0].cols & -2, neuEbenen[0].rows & -2));
    neuEbenen[1] = neuEbenen[1](Rect(0, 0, neuEbenen[1].cols & -2, neuEbenen[1].rows & -2));

    int cx1 = neuEbenen[0].cols>>1;
    int cy1 = neuEbenen[0].rows>>1;
    Mat tmp1;

    tmp1.create(neuEbenen[0].size(),neuEbenen[0].type());
    neuEbenen[0](Rect(0, 0, cx1, cy1)).copyTo(tmp1(Rect(cx1, cy1, cx1, cy1)));
    neuEbenen[0](Rect(cx1, cy1, cx1, cy1)).copyTo(tmp1(Rect(0, 0, cx1, cy1)));
    neuEbenen[0](Rect(cx1, 0, cx1, cy1)).copyTo(tmp1(Rect(0, cy1, cx1, cy1)));
    neuEbenen[0](Rect(0, cy1, cx1, cy1)).copyTo(tmp1(Rect(cx1, 0, cx1, cy1)));
    neuEbenen[0]=tmp1;

    int cx2 = neuEbenen[1].cols>>1;
    int cy2 = neuEbenen[1].rows>>1;
    Mat tmp2;

    tmp2.create(neuEbenen[1].size(),neuEbenen[1].type());
    neuEbenen[1](Rect(0, 0, cx2, cy2)).copyTo(tmp2(Rect(cx2, cy2, cx2, cy2)));
    neuEbenen[1](Rect(cx2, cy2, cx2, cy2)).copyTo(tmp2(Rect(0, 0, cx2, cy2)));
    neuEbenen[1](Rect(cx2, 0, cx2, cy2)).copyTo(tmp2(Rect(0, cy2, cx2, cy2)));
    neuEbenen[1](Rect(0, cy2, cx2, cy2)).copyTo(tmp2(Rect(cx2, 0, cx2, cy2)));
    neuEbenen[1]=tmp2;
    neuEbenen[0]/=float(M*N);
    neuEbenen[1]/=float(M*N);
    ebenen[0]=neuEbenen[0].clone();
    ebenen[1]=neuEbenen[1].clone();
    Mag.zeros(ebenen[0].rows,ebenen[0].cols,CV_32F);
    Phase.zeros(ebenen[0].rows,ebenen[0].cols,CV_32F);
    cv::cartToPolar(ebenen[0],ebenen[1],Mag,Phase);
    Mat maske(Mag.cols, Mag.rows, CV_32F, Scalar(0));

    int cx = Mag.cols>>1;
    int cy = Mag.rows>>1;

    maske=1;
    cv::circle(maske,cv::Point(cx,cy),40,CV_RGB(0,0,0),-1);
    maske=1-maske;
    cv::multiply(Mag,maske,Mag);
    cv::multiply(Phase,maske,Phase);
    ebenen[0].create(Mag.rows,Mag.cols,CV_32F);
    ebenen[1].create(Mag.rows,Mag.cols,CV_32F);
    cv::polarToCart(Mag,Phase,ebenen[0],ebenen[1]);

    int cx3 = ebenen[0].cols>>1;
    int cy3 = ebenen[0].rows>>1;
    Mat tmp3;

    tmp3.create(ebenen[0].size(),ebenen[0].type());
    ebenen[0](Rect(0, 0, cx3, cy3)).copyTo(tmp3(Rect(cx3, cy3, cx3, cy3)));
    ebenen[0](Rect(cx3, cy3, cx3, cy3)).copyTo(tmp3(Rect(0, 0, cx3, cy3)));
    ebenen[0](Rect(cx3, 0, cx3, cy3)).copyTo(tmp3(Rect(0, cy3, cx3, cy3)));
    ebenen[0](Rect(0, cy3, cx3, cy3)).copyTo(tmp3(Rect(cx3, 0, cx3, cy3)));
    ebenen[0]=tmp3;

    int cx4 = ebenen[1].cols>>1;
    int cy4 = ebenen[1].rows>>1;
    Mat tmp4;

    tmp4.create(ebenen[1].size(),ebenen[1].type());
    ebenen[1](Rect(0, 0, cx4, cy4)).copyTo(tmp4(Rect(cx4, cy4, cx4, cy4)));
    ebenen[1](Rect(cx4, cy4, cx4, cy4)).copyTo(tmp4(Rect(0, 0, cx4, cy4)));
    ebenen[1](Rect(cx4, 0, cx4, cy4)).copyTo(tmp4(Rect(0, cy4, cx4, cy4)));
    ebenen[1](Rect(0, cy4, cx4, cy4)).copyTo(tmp4(Rect(cx4, 0, cx4, cy4)));
    ebenen[1]=tmp4;

    Mat neuKomplexBild;

    merge(ebenen, 2, neuKomplexBild);

    idft(neuKomplexBild, neuKomplexBild);
    split(neuKomplexBild, ebenen);
    normalize(ebenen[0], bild, 0, 1, CV_MINMAX);

    Mat LogMag;

    LogMag.zeros(Mag.rows,Mag.cols,CV_32F);
    LogMag=(Mag+1);
    cv::log(LogMag,LogMag);

    imshow("Logarithmus der Amplitude", LogMag);
    imshow("Filterergebniss", bild);
    cvWaitKey(0);
    return 0;
}
