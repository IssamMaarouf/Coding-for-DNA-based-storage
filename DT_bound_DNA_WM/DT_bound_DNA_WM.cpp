///////////////////////////////////////////////////////////////////////////
// Issam Maarouf
// DT bound calculation for WM inner code
// February 2021
///////////////////////////////////////////////////////////////////////////

//Include
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

////////////////////////////////////////
//Simulation Parameters
////////////////////////////////////////

//Parameters Encoders
//Parameters LDPC code
#define Def_NumSeq                   1 //Number of sequences (DNA strand reads)

#define Def_NBitsPerSymbolIn         4 //Number of bits per symbol (input of inner code)
#define GF                           16 // Galois Feild order
#define Def_NBitsPerSymbolOut        2 //Number of bits per symbol (output of inner code)

#define Def_kInner               120 //Inner code information length (interleaver size)

#define Def_nWM                      8 //Codeword length watermark code
#define Def_kWM                      4 //Information word length watermark code

#define Def_nInner            (Def_kInner*Def_nWM/Def_NBitsPerSymbolOut) //Length interleaver in symbols

#define output_type					  0 //0 : singular otuput (debugging) 
                                        //1 : output to parralelization file

#define Def_MinNumberDTboundis1      10000 // Required number of DT bound is equal to 1


//Inner codebook choice
#define inner_codebook_design         3 //0 : use the Davey-MacKay construction
                                        //1 : use construction that minimizes the change probability (CP)
                                        //2 : manually input the codebook
                                        //3 : manually input n number of codebooks
                                        //4 : VT code
                                        //5 : use a random codebook


#define a_VT                          1 // a parameter of VT code.

#define inner_codebook_numb              4 // number of inner codebooks

#define added_sequence_type           3 //0 : use a random sequnce
                                        //1 : no random sequence
                                        //2 : use n best substrings
                                        //3 : choose between n codeboos

#define added_sequence_numb             4 //Number of substrings you can choose to add to the transmitted data
//Inner codebook choice


#define bound_type                      0 //0 : DT/RCUs bound
                                          //1 : Converse bound

#define RCUs_interval                  0.0 // Max s vlue for RCUs/Converse bound
#define RCUs_intervaldelta             1.0 // Quantization of RCUs/Converse interval
#define RCUs_intervalpoints          int(RCUs_interval / RCUs_intervaldelta) + 1 //Number of s points to be computed

#define f                       0.3125 //Mean density of sparse vectors


//Parameters ID channel
#define Ins_max                      2 //Maxiumum number of consecutive insertions as considered by the decoder
#define Del_max                      1 //Maxiumum number of consecutive deletions  as considered by the decoder


//EXIT Charts definitions
#define Def_NumBlocks           100000
#define Def_MIPoints            20

double  H1 = 0.3073;          // Parameters for the computation of J function
double  H2 = 0.8935;          // Parameters for the computation of J function
double  H3 = 1.1064;          // Parameters for the computation of J function
int     BitsPerSymbol;


#define Def_InsDelProbIni         0.10// Initial Insertion/Deletion probability
#define Def_SubsProb              0.00 // Substitution Probability
#define Def_InsDelProbDelta       0.01 // decrease in bit cross-over probability for each simulation point
#define Def_NUM_POINTS              7 // //Number of points to be simulated


#define Def_vc_unInter              2344 // Initial seed generation of an integer (between 0 and a given integer) uniformly at random (used
// when generating the error pattern)
#define Def_vc_unInterH                2000 // Initial seed generation of an integer (between 0 and a given integer) uniformly at random (used
                                         // when generating the parity-check matrix)
#define Def_vcunstart                  1 // Initial seed for uniform distribution (between 0 and 1) NOT used here
#define Def_lstart					   1 // Initial seed for generator of random binary sequences NOT used here

#define MAXS_H_INTERVAL			    0.01 //
#define MAXS_H_NUMSAMPLES			1000 //

#define Joint_Decoding              1// 0: Separate decoding of multiple sequences
                                     // 1: Joint decoding of multiple sequences 

////////////////////////////////////////
//END Simulation Parameters
////////////////////////////////////////

double jaclog_lookup[MAXS_H_NUMSAMPLES];

int LookUpOnesSparse[4096];
int LookUpOnes[GF];
int LookUpSparse[inner_codebook_numb][GF] = { 0 };
int watermark_strings[added_sequence_numb][Def_nWM] = { 0 };
int receivedword[Def_NumSeq][2 * Def_nInner];

int codeword_ref_CC[Def_nInner] = { 0 };

int perm_sparse[4096];
int perm_code[Def_nInner];


int add_tableDNA[4][4] = { {0, 1, 2, 3 },
                           {1, 0, 3, 2 },
                           {2, 3, 0, 1 },
                           {3, 2, 1, 0 } };


int state_mapper[300000][Def_NumSeq] = { 0 };

double Forward[300000][Def_kInner + 1] = { 0 };

//vector<vector<vector<double>>> Middle(pow(2, Def_nWM), vector<vector<double>>(Def_nWM / 2 * (Ins_max + 1) + 1, vector<double>(pow(4, Def_nWM / 2 * (Ins_max + 1)), -1)));

//vector<vector<vector<double>>> Middle(pow(2, Def_nWM), vector<vector<double>>(Def_nWM /*/ 2*/* (Ins_max + 1) + 1, vector<double>(pow(2, Def_nWM /*/ 2*/* (Ins_max + 1)), -1)));

vector<vector<vector<vector<double>>>> Middle(Def_NumSeq, vector<vector<vector<double>>>(Def_nInner * 2 + 1, vector<vector<double>>(pow(2, Def_nWM), vector<double>(Def_nWM / 2 * (Ins_max + 1) + 1))));

int decide_transm_int[GF][Def_kInner];

/*************************************************************************/
/**************************   GENREAL VARIABLES   ************************/
/*************************************************************************/

double F_pq[Def_nInner + 2][2 * Def_nInner + 1];

//DT bound Variables
double DTbound[100][2000];
double DTboundAv[100][2000];
double Inf_density[Def_NumBlocks];
int numb_blocks[RCUs_intervalpoints] = { 0 };

int    interleaver[Def_nInner + 2];

typedef struct
{
    int nedge;
    int trellis[9000000][2 * Def_NumSeq + 2];
} TrellisStruct;
TrellisStruct WatermarkCodeTrellis;


unsigned long int l;                  /* 32 celle LFSR */
unsigned long int l_start;            /* valore iniziale LFSR */

long int vc_un;                       /* variabile intera 31 bit per    */
long int vc_unInterH;                       /* variabile intera 31 bit per    */
long int vc_unInter;                       /* variabile intera 31 bit per    */
                                      /* la variabile uniforme */
long int vc_un_start;                 /* valore iniziale */
long int vc_unch;

unsigned long int l_orig;                  /* 32 celle LFSR */
unsigned long int l_start_orig;            /* valore iniziale LFSR */

long int vc_un_orig;                       /* variabile intera 31 bit per    */
long int vc_unInterH_orig;                       /* variabile intera 31 bit per    */
long int vc_unInter_orig;                       /* variabile intera 31 bit per    */
                                      /* la variabile uniforme */
long int vc_un_start_orig;                 /* valore iniziale */
long int vc_unch_orig;


int DTboundis1;
int MinNumberDTboundis1;

double P_ins;                         //Probability of insertion
double P_del;                         //Probability of deletion
double P_subs;                        //Probability of substitution
double P_trans;                       //Probability of transmission
int drift_max;
int states_max;
int s_points;
int max_s;

int ncheck;                           /* periodo risultati intermedi */



//Variables SCC
long int seed;





/*************************************************************************/
/******************************   LFSR   *********************************/
/*************************************************************************/
int lfsr()
{
    int b, l2;

    b = ((l & (1LL << 31)) >> 31);

    l2 = (l & 1) ^ ((l & (1LL << 1)) >> 1) ^ ((l & (1LL << 21)) >> 21) ^ b;

    l = (l << 1) | l2;

    return(b);
}



/*************************************************************************/
/****** unif: generate a uniform distributed random variable *************/
/*************************************************************************/
double unif(void)
{
    long s;
    long mm = 0x7FFFFFFF, a = 16807, q = 127773, r = 2836;

    s = vc_un / q;
    vc_un = a * (vc_un - q * s) - r * s;
    if (vc_un <= 0) vc_un += mm;

    return ((double)vc_un / mm);
}

double unif_ch(void)
{
    long s;
    long mm = 0x7FFFFFFF, a = 16807, q = 127773, r = 2836;

    s = vc_unch / q;
    vc_unch = a * (vc_unch - q * s) - r * s;
    if (vc_unch <= 0) vc_unch += mm;

    return ((double)vc_unch / mm);
}

int unif_int()
{
    long s;
    const long mm = 0x7FFFFFFF, a = 16807, q = 127773, r = 2836;
    s = vc_unInter / q; //vc_unInter;
    vc_unInter = a * (vc_unInter - q * s) - r * s;
    if (vc_unInter <= 0) vc_unInter += mm;
    return (vc_unInter);
}

int unif_intH()
{
    long s;
    const long mm = 0x7FFFFFFF, a = 16807, q = 127773, r = 2836;
    s = vc_unInterH / q;
    vc_unInterH = a * (vc_unInterH - q * s) - r * s;
    if (vc_unInterH <= 0) vc_unInterH += mm;
    return (vc_unInterH);
}

double Gaussian()
{
    static double f1, f2, f3, a;
    static double x2;
    static int isready = 0;
    if (isready)
    {
        isready = 0;
        return x2;
    }
    else
    {

        f3 = 100.0;
        while (f3 > 1.0)
        {
            f1 = 2.0 * unif() - 1.0;
            f2 = 2.0 * unif() - 1.0;
            f3 = f1 * f1 + f2 * f2;
        }
        a = sqrt(-2.0 * log(f3) / f3);

        x2 = f2 * a;
        isready = 1;
        return (f1 * a);
    }
}


int willIstop()
{
    if (DTboundis1 < MinNumberDTboundis1) return 0;

    return 1;
}

/*************************************************************************/
/***************   Initialize variables and simulation   *****************/
/*************************************************************************/
void initialize_variables()
{
    int mI, i, ipick, temp;
    int GF_sparse = pow(2, Def_nWM);

    ifstream inFile;


    BitsPerSymbol = (int)(log(GF) / log(2.));

    for (mI = 0; mI <= Def_NUM_POINTS; mI++)
    {
        for (i = 0; i < RCUs_intervalpoints; i++)
        {
            DTbound[i][mI] = 0.;
            DTboundAv[i][mI] = 0.;
        }
    }

    l_start = l_start_orig;
    l = l_orig;
    vc_un = vc_un_orig;
    vc_unInter = vc_unInter_orig;
    vc_unInterH = vc_unInterH_orig;
    vc_unch = vc_unch_orig;
    //printf("Number of simulation points %d \n",numero_sim);

    s_points = RCUs_intervalpoints;

    for (i = 0; i < GF_sparse; i++)
        perm_sparse[i] = i;

    //random inner codebook generation (NOT ALWAYS USED)
    for (i = 0; i < GF; i++)
    {
        ipick = i + unif_int() % (GF_sparse - i);
        temp = perm_sparse[i];
        perm_sparse[i] = perm_sparse[ipick];
        perm_sparse[ipick] = temp;
    }

    for (i = 0; i < MAXS_H_NUMSAMPLES; i++)
    {
        jaclog_lookup[i] = log(1 + exp(-i * MAXS_H_INTERVAL));
    }

}


/*
/////////////////////////////
/// Calculate receiver metric
/////////////////////////////
double Receiver_metric(int* temp_recieved, int* temp_trans, int length1, int length2)
{
    int i, j;
    double F_pq[20][20];

    F_pq[0][0] = 1;

    for (j = 1; j <= length1; j++)
        F_pq[0][j] = 0.25 * P_ins * F_pq[0][j - 1];

    for (i = 1; i <= length2; i++)
        F_pq[i][0] = P_del * F_pq[i - 1][0];

    for (i = 1; i <= length2; i++)
    {
        for (j = 1; j <= length1; j++)
        {
            if (i == length2)
            {
                if (temp_trans[i - 1] == temp_recieved[j - 1])
                    F_pq[i][j] = P_del * (F_pq[i - 1][j] * (i - 1 >= 0)) + P_trans * (1 - P_subs) * (F_pq[i - 1][j - 1] * (i - 1 >= 0 && j - 1 >= 0));
                else
                    F_pq[i][j] = P_del * (F_pq[i - 1][j] * (i - 1 >= 0)) + P_trans * P_subs / 3 * (F_pq[i - 1][j - 1] * (i - 1 >= 0 && j - 1 >= 0));
            }
            else
            {
                if (temp_trans[i - 1] == temp_recieved[j - 1])
                    F_pq[i][j] = 0.25 * P_ins * (F_pq[i][j - 1] * (j - 1 >= 0)) + P_del * (F_pq[i - 1][j] * (i - 1 >= 0)) + P_trans * (1 - P_subs) * (F_pq[i - 1][j - 1] * (i - 1 >= 0 && j - 1 >= 0));
                else
                    F_pq[i][j] = 0.25 * P_ins * (F_pq[i][j - 1] * (j - 1 >= 0)) + P_del * (F_pq[i - 1][j] * (i - 1 >= 0)) + P_trans * P_subs / 3 * (F_pq[i - 1][j - 1] * (i - 1 >= 0 && j - 1 >= 0));
            }
        }
    }

    return(F_pq[length2][length1]);
}
/////////////////////////////
/// Calculate receiver metric
/////////////////////////////
*/



//////////////////////////////
/// Look up Max(x,y)* function 
//////////////////////////////
double max_starLU(double x, double y) {

    int i = round(abs(x - y) / MAXS_H_INTERVAL);

    if (i >= MAXS_H_NUMSAMPLES || x == -INFINITY || y == -INFINITY)
        return (x > y ? x : y);
    else
        return (x > y ? x : y) + jaclog_lookup[i];
}
//////////////////////////////
/// Look up Max(x,y)* function 
//////////////////////////////

/////////////////////////
/// Max(x,y)* function 
/////////////////////////
double max_star(double x, double y)
{
    double x_star, y_star;
    double d = abs(x - y);

    if (x == -INFINITY || y == -INFINITY)
        return (x > y ? x : y);

    return (x > y ? x : y) + log(1 + exp(-d));
}
/////////////////////////
/// Max(x,y)* function 
/////////////////////////



/////////////////////////////
/// Calculate receiver metric
/////////////////////////////
double Receiver_metric(int* temp_recieved, int* temp_trans, int length1, int length2)
{
    int i, j;

    double P_EFF = 0;

    for (i = 0; i <= length2; i++)
        for (j = 0; j <= length1; j++)
            F_pq[i][j] = 0;


    P_EFF = P_subs;


    F_pq[0][0] = 1;

    for (j = 1; j <= length1; j++)
        F_pq[0][j] = 0.25 * P_ins * F_pq[0][j - 1];

    for (i = 1; i <= length2; i++)
        F_pq[i][0] = P_del * F_pq[i - 1][0];

    for (i = 1; i <= length2; i++)
    {
        for (j = 1; j <= length1; j++)
        {
            if (i == length2)
            {
                if (temp_trans[i - 1] == temp_recieved[j - 1])
                    F_pq[i][j] = P_del * (F_pq[i - 1][j] * (i - 1 >= 0)) + P_trans * (1 - P_EFF) * (F_pq[i - 1][j - 1] * (i - 1 >= 0 && j - 1 >= 0));
                else
                    F_pq[i][j] = P_del * (F_pq[i - 1][j] * (i - 1 >= 0)) + P_trans * P_EFF / 3 * (F_pq[i - 1][j - 1] * (i - 1 >= 0 && j - 1 >= 0));
            }
            else
            {
                if (temp_trans[i - 1] == temp_recieved[j - 1])
                    F_pq[i][j] = 0.25 * P_ins * (F_pq[i][j - 1] * (j - 1 >= 0)) + P_del * (F_pq[i - 1][j] * (i - 1 >= 0)) + P_trans * (1 - P_EFF) * (F_pq[i - 1][j - 1] * (i - 1 >= 0 && j - 1 >= 0));
                else
                    F_pq[i][j] = 0.25 * P_ins * (F_pq[i][j - 1] * (j - 1 >= 0)) + P_del * (F_pq[i - 1][j] * (i - 1 >= 0)) + P_trans * P_EFF / 3 * (F_pq[i - 1][j - 1] * (i - 1 >= 0 && j - 1 >= 0));
            }
        }
    }

    return(F_pq[length2][length1]);
}
/////////////////////////////
/// Calculate receiver metric
/////////////////////////////


/////////////////////////////
/// Calculate receiver metric
/////////////////////////////
double Receiver_metric_log(int* temp_recieved, int* temp_trans, int length1, int length2)
{
    int i, j;

    for (i = 0; i <= length2; i++)
        for (j = 0; j <= length1; j++)
            F_pq[i][j] = -INFINITY;

    double P_EFF = 0;

    P_EFF = P_subs;


    F_pq[0][0] = 0;

    for (j = 1; j <= length1; j++)
        F_pq[0][j] = log(0.25) + log(P_ins) + F_pq[0][j - 1];

    for (i = 1; i <= length2; i++)
        F_pq[i][0] = log(P_del) + F_pq[i - 1][0];

    for (i = 1; i <= length2; i++)
    {
        for (j = 1; j <= length1; j++)
        {
            if (i == length2)
            {
                F_pq[i][j] = max_star(log(P_del) + (F_pq[i - 1][j] + log(i - 1 >= 0)), F_pq[i][j]);

                if (temp_trans[i - 1] == temp_recieved[j - 1])
                    F_pq[i][j] = max_star(log(P_trans) + log(1 - P_EFF) + (F_pq[i - 1][j - 1] + log(i - 1 >= 0 && j - 1 >= 0)), F_pq[i][j]);
                else
                    F_pq[i][j] = max_star(log(P_trans) + log(P_EFF / 3) + (F_pq[i - 1][j - 1] + log(i - 1 >= 0 && j - 1 >= 0)), F_pq[i][j]);
            }
            else
            {

                F_pq[i][j] = max_star(log(0.25) + log(P_ins) + (F_pq[i][j - 1] + log(j - 1 >= 0)), F_pq[i][j]);

                F_pq[i][j] = max_star(log(P_del) + (F_pq[i - 1][j] + log(i - 1 >= 0)), F_pq[i][j]);

                if (temp_trans[i - 1] == temp_recieved[j - 1])
                    F_pq[i][j] = max_star(log(P_trans) + log(1 - P_EFF) + (F_pq[i - 1][j - 1] + log(i - 1 >= 0 && j - 1 >= 0)), F_pq[i][j]);
                else
                    F_pq[i][j] = max_star(log(P_trans) + log(P_EFF / 3) + (F_pq[i - 1][j - 1] + log(i - 1 >= 0 && j - 1 >= 0)), F_pq[i][j]);
            }
        }
    }

    return(F_pq[length2][length1]);
}
/////////////////////////////
/// Calculate receiver metric
/////////////////////////////


//////////////////////////////////
/// Compute P(Y) using the lattice
//////////////////////////////////
double logPY(int* temp_recieved, int length1, int length2) {

    int i, j;

    for (i = 0; i <= length2; i++)
        for (j = 0; j <= length1; j++)
            F_pq[i][j] = -INFINITY;


    // starting point
    F_pq[0][0] = 0;

    // iterative computation of values
    for (int p = 0; p < length2 + 1; p++) {
        for (int r = 0; r < length1 + 1; r++) {
            if (p < length2 && r < length1)
                F_pq[p + 1][r + 1] = max_star(F_pq[p + 1][r + 1], log(P_trans / 4) + F_pq[p][r]);      // substitutions
            if (r < length1)
                F_pq[p][r + 1] = max_star(F_pq[p][r + 1], log(P_ins / 4) + F_pq[p][r]);          // insertions
            if (p < length2)
                F_pq[p + 1][r] = max_star(F_pq[p + 1][r], log(P_del) + F_pq[p][r]);              // deletions

        }
    }

    // fix insertions in the last row
    for (int p = 0; p < length2 + 1; p++) {
        for (int r = length1; r > 0; r--) {
            if (F_pq[p][r] != -INFINITY)
                F_pq[p][r] = F_pq[p][r] + log(1 - P_ins / 4 * exp(F_pq[p][r - 1] - F_pq[p][r]));
        }
    }

    return(F_pq[length2][length1]);
}
//////////////////////////////////
/// Compute P(Y) using the lattice
//////////////////////////////////



//Mapp integer to vector over GF(q)
void int2vec_q(int* vect, int integ, int vect_length, int q)
{
    int i, j;
    int cnt = 0;

    for (i = 0; i < vect_length; i++)
    {
        vect[i] = 0;
    }

    while (integ > 0)
    {
        vect[cnt++] = integ % q;
        integ /= q;
    }

}


//Mapp vector to integer over GF(q)
void vec2int_q(int* vect, int& integ, int vect_length, int q)
{
    int i;

    integ = 0;
    for (i = 0; i < vect_length; i++)
    {
        integ += pow(q, i) * vect[i];
    }

}


/*
////////////////////////////////////////
/// Pre-compute branch/receiver metrics
////////////////////////////////////////
void compute_BranchMetrics(int nw, int q, double s_point)
{
    int i, j, ii, jj;
    int max_lengthY = nw * (Ins_max + 1);

    int* x = new int[nw]; //transmitted symbol
    int* y = new int[max_lengthY]; //Received sequence

    int y_int; //decimal value of received sequence

    for (i = 0; i < pow(q, nw); i++)
    {//loop over all possible transmitted symbols X

        int2vec_q(x, i, nw, q);

        for (j = 0; j <= max_lengthY; j++)
        {//loop over all possible lengths of received sequence Y
            for (ii = 0; ii < pow(q, j); ii++)
            {//Loop over s values for RCUs bound

                int2vec_q(y, ii, max_lengthY, q); //convert int reprsentation of Y to a vect representation

                Middle[i][j][ii] = log(pow(Receiver_metric(y, x, j, nw), s_point));
            }
        }
    }

    delete[] x;
    delete[] y;
}
////////////////////////////////////////
/// Pre-compute branch/receiver metrics
////////////////////////////////////////
*/


void Build_inner_codetrellis(int nw, int states_max)
{
    int count = 0;
    int cond = 0;
    int rep = Def_NumSeq;

    if (Joint_Decoding == 0)
        rep = 1;

    for (int x_2 = 0; x_2 < states_max; x_2++)
    {
        for (int x_1 = 0; x_1 < states_max; x_1++)
        {
            for (int i = 0; i < rep; i++)
            {
                if (state_mapper[x_1][i] - state_mapper[x_2][i] > nw || state_mapper[x_2][i] - state_mapper[x_1][i] > Ins_max * (nw))
                    cond++;
            }

            if (cond > 0)
            {
                cond = 0;
                continue;
            }
            else
            {
                cond = 0;
                WatermarkCodeTrellis.trellis[count][0] = x_1;
                for (int i = 0; i < rep; i++)
                    WatermarkCodeTrellis.trellis[count][i + 1] = state_mapper[x_1][i];

                WatermarkCodeTrellis.trellis[count][rep + 1] = x_2;
                for (int i = 0; i < rep; i++)
                    WatermarkCodeTrellis.trellis[count][rep + 1 + i + 1] = state_mapper[x_2][i];

                count++;
            }
        }
    }

    WatermarkCodeTrellis.nedge = count;
}
/////////////////////////////
/// Build inner code trellis
/////////////////////////////


void initialize_sim(int& kInner, int nWM, int sim, int& nInner, double s_point)
{
    int i, j;
    int rep = Def_NumSeq;

    l_start = l_start_orig;
    l = l_orig;
    vc_un = vc_un_orig;
    vc_unInter = vc_unInter_orig;
    vc_unInterH = vc_unInterH_orig;
    vc_unch = vc_unch_orig;

    //kInner = Def_kInner * Def_InsDelProbDelta * (sim + 1);
    //nInner = kInner * nWM / 2;
    P_ins = Def_InsDelProbIni - sim * Def_InsDelProbDelta;
    P_del = Def_InsDelProbIni - sim * Def_InsDelProbDelta;
    //P_del = 0;
    P_trans = 1 - P_ins - P_del;
    P_subs = Def_SubsProb;

    drift_max = 5 * sqrt(nInner * P_del / (1 - P_del));
    if (Joint_Decoding == 0)
    {
        states_max = 2 * drift_max + 1;
        rep = 1;
    }
    else
        states_max = pow(2 * drift_max + 1, rep);

    for (i = 0; i < states_max; i++)
    {
        for (j = 0; j < rep; j++)
        {
            state_mapper[i][j] = -drift_max + (int(i / pow(2 * drift_max + 1, rep - 1 - j))) % (2 * drift_max + 1);
        }
    }

    //Pre-compute branch/receiver metrics P(Y|X) for all X and Y
    //compute_BranchMetrics(nWM / 2, 4, s_point);

    //compute_BranchMetrics(nWM, 2);

    Build_inner_codetrellis(nWM / 2, states_max);

    //Build_inner_codetrellis(nWM, drift_max);
}



/////////////////////////////////
/// Calculate change probability
/////////////////////////////////
void Change_Probability(int nWM, double& P_C, int codebook[GF])
{
    int i, j, ii, jj;

    int str1[Def_nWM] = { 0 };
    int str2[Def_nWM] = { 0 };

    for (i = 0; i < GF; i++)
    {
        for (ii = 0; ii < Def_nWM; ii++)
            str1[ii] = (codebook[i] >> ii) & 1LL;

        for (j = 0; j < GF; j++)
        {
            if (j != i)
            {
                for (jj = 0; jj < Def_nWM; jj++)
                    str2[jj] = (codebook[j] >> jj) & 1LL;

                P_C += Receiver_metric(str1, str2, Def_nWM, Def_nWM);
            }
            else
                continue;
        }
    }

    P_C /= (GF * (GF - 1));
}
/////////////////////////////////
/// Calculate change probability
/////////////////////////////////


/////////////////////////////
/// Build inner codebook
/////////////////////////////
void build_inner_codebook(int nWM, int design)
{
    int i, j, k, ii, jj, ipick;
    double sum = 0;
    int weight = 1;
    int temp;
    int GF_sparse = pow(2, Def_nWM);
    int check_count[Def_nWM + 1] = { 0 };

    //Compute the # of 1's in each in vector in 2^(nw)
    for (i = 0; i < GF_sparse; i++)
    {
        int count = 0;
        int numb = i;

        while (numb != 0)
        {
            numb = numb & (numb - 1);
            count++;
        }

        check_count[count]++;

        LookUpOnesSparse[i] = count;
    }


    if (design == 0)
    {
        for (i = 1; i < GF;)
        {
            j = check_count[weight];

            while (j > 0 && i < GF)
            {
                for (ii = 0; ii < GF_sparse; ii++)
                {
                    if (LookUpOnesSparse[ii] == weight)
                    {
                        LookUpSparse[0][i] = ii;
                        i++;
                        j--;
                        if (i >= GF)
                            break;
                    }
                }
            }
            weight++;
        }

        /*
        for (i = 0; i < GF; i++)
            LookUpSparse[0][i] = perm_sparse[i];
        */
    }
    else if (design == 1)
    {
        double T = 1000;
        double alpha = 0.99;
        double P_C = 0;
        double P_C_prime = 0;
        double P_C_init = 0;
        int pertub_c;
        int pertub_c_temp;
        int pertub_b;
        int E_drops_max = 50;
        int E_drops = 0;
        int iter_max = 10000;
        int iter = 0;
        double E_delta1 = 0;
        double E_delta2 = 0;
        int E_unchanged = 0;
        int codeword[Def_nWM] = { 0 };
        int numb;

        for (i = 0; i < GF_sparse; i++)
            perm_sparse[i] = i;

        for (i = 0; i < GF; i++)
        {
            ipick = i + unif_int() % (GF_sparse - i);
            temp = perm_sparse[i];
            perm_sparse[i] = perm_sparse[ipick];
            perm_sparse[ipick] = temp;
        }

        for (i = 0; i < GF; i++)
            LookUpSparse[0][i] = perm_sparse[i];

        //Calculate the energy of the chosen code C
        Change_Probability(nWM, P_C, LookUpSparse[0]);

        P_C_init = P_C;

        while (E_unchanged < 10)
        {
            while (iter < iter_max && E_drops < E_drops_max)
            {
                numb = 0;

                pertub_c = unif_int() % GF;
                pertub_c_temp = LookUpSparse[0][pertub_c];
                pertub_b = unif_int() % nWM;

                for (i = 0; i < nWM; i++)
                    codeword[i] = (LookUpSparse[0][pertub_c] >> i) & 1LL;

                codeword[pertub_b] = codeword[pertub_b] ^ 1;

                for (i = 0; i < nWM; i++)
                    numb += codeword[i] << i;

                //Change from C to C'
                LookUpSparse[0][pertub_c] = numb;

                //Calculate the energy of the chosen code C'
                Change_Probability(nWM, P_C_prime, LookUpSparse[0]);

                E_delta2 = P_C_prime - P_C;

                if (E_delta2 < 0 || unif() > exp(-E_delta2 / T))
                {
                    P_C = P_C_prime;
                    E_drops++;
                    iter++;
                    //E_delta1 = E_delta2;
                }
                else
                {
                    LookUpSparse[0][pertub_c] = pertub_c_temp;
                    iter++;
                    E_delta2 = E_delta1;
                }
            }

            T = alpha * T;
            iter = 0;
            E_drops = 0;

            if (P_C == P_C_init)
                E_unchanged++;

            P_C_init = P_C;
        }

        //Calculate the energy of the chosen code C
        Change_Probability(nWM, P_C, LookUpSparse[0]);
    }
    else if (design == 2)
    {
        //int manual_codebook[GF] = { 0,60,26,38,41,21,51,15 }; //HD 1
        //int manual_codebook[GF] = { 32,24,42,6,9,57,27,39 }; //LD 1
        //int manual_codebook[GF] = { 52,12,2,58,33,25,23,47 }; //HD 2
        //int manual_codebook[GF] = { 0,56,12,42,33,19,15,63 }; //LD 2

        //int manual_codebook[GF] = { 33,18,12,52,30,45,11,39 }; //VT_0(6) 

        int manual_codebook[GF] = { 0,1,2,3 };

        for (i = 0; i < GF; i++)
            LookUpSparse[0][i] = manual_codebook[i];
    }
    else if (design == 3)
    {

        /*
            int manual_n_codebook[inner_codebook_numb][GF] = { {0,56,12,34,21,61,3,39 },
                                    {48,20,6,54,1,37,23,63},
                                    {32,60,50,10,17,43,7,31},
                                    {24,36,2,14,62,57,35,27},
                                    {8,52,38,30,33,25,11,59} };
          */
          /*
          int manual_n_codebook[inner_codebook_numb][GF] = { {0,44,19,57,18,42,7,63 },
                          {0,28,19,33,53,42,11,63} };
          */
          /*
          int manual_n_codebook[inner_codebook_numb][GF] = { {0,44,21,57,18,42,7,63 },
          {0,28,21,33,54,42,11,63} };
          */
          /*
                  int manual_n_codebook[inner_codebook_numb][GF] = { {0,160,236,17,85,245,185,98,38,170,250,115,55,11,175,255},
          {64,240,152,33,165,117,205,130,202,90,222,19,171,59,95,191} };
          */

        int manual_n_codebook[inner_codebook_numb][GF] = { {0,160,236,17,85,245,185,98,38,170,250,115,55,11,175,255},
        {64,240,152,33,165,117,205,130,202,90,222,19,171,59,95,191},
        {48,84,184,220,65,137,237,10,218,170,206,131,155,95,47,255},
        {192,160,188,81,5,245,185,102,10,234,206,67,55,139,123,255} };

        //int manual_n_codebook[inner_codebook_numb][GF] = { 0 };
        //int manual_n_codebook[inner_codebook_numb][GF] = { 0,56,12,34,21,61,3,39 };

        for (i = 0; i < inner_codebook_numb; i++)
            for (j = 0; j < GF; j++)
                LookUpSparse[i][j] = manual_n_codebook[i][j];
    }
    else if (design == 4)
    {
        int codeword[Def_nWM / 2] = { 0 };
        int sum1 = 0, sum2 = 0, cnt = 0;
        int VT_codebook[200] = { 0 };
        //int As[Def_nWM / 2 - 1] = { 0 };
        int As[Def_nWM / 2] = { 0 };

        for (i = 0; i < pow(4, nWM / 2); i++)
        {
            int2vec_q(codeword, i, nWM / 2, 4);

            for (j = 0; j < nWM / 2 - 1; j++)
                As[j] = 0;
            for (j = 0; j < nWM / 2 - 1; j++)
                if (codeword[j + 1] >= codeword[j])
                    As[j] = 1;

            sum1 = 0;
            sum2 = 0;

            for (j = 0; j < nWM / 2; j++)
                sum1 += codeword[j];

            for (j = 0; j < nWM / 2 - 1; j++)
                sum2 += (j + 1) * As[j];

            sum1 %= 4;
            sum2 %= (nWM / 2 + 1);

            if (sum1 == 0 && sum2 == a_VT)
                VT_codebook[cnt++] = i;
        }

        for (j = 0; j < GF; j++)
            for (i = 0; i < inner_codebook_numb; i++)
                //LookUpSparse[i][j] = VT_codebook[i * GF + j];
                LookUpSparse[i ^ j % 2][j] = VT_codebook[i * GF + j];

        int overlap = 0;
        int min_overlap = 64;
        int permute[1000];
        int coodebook_min_overlap[GF];
        int iter = 0;

        for (i = 2; i < inner_codebook_numb; i++)
        {
            min_overlap = 64;
            for (iter = 0; iter < 100; iter++)
            {
                overlap = 0;

                for (j = 0; j < cnt; j++)
                    permute[j] = j;

                for (j = 0; j < GF; j++)
                {
                    ipick = j + unif_int() % (cnt - j);
                    temp = permute[j];
                    permute[j] = permute[ipick];
                    permute[ipick] = temp;
                }

                for (j = 0; j < GF; j++)
                {
                    coodebook_min_overlap[j] = VT_codebook[permute[j]];
                }

                for (ii = 0; ii < i; ii++)
                {
                    for (j = 0; j < GF; j++)
                    {
                        if (LookUpSparse[ii][j] == coodebook_min_overlap[j])
                        {
                            overlap++;
                        }
                    }
                }

                if (overlap < min_overlap)
                {
                    min_overlap = overlap;
                    for (j = 0; j < GF; j++)
                        LookUpSparse[i][j] = coodebook_min_overlap[j];
                }
            }
        }
    }
    else
    {
        for (i = 0; i < GF; i++)
            LookUpSparse[0][i] = perm_sparse[i];
    }
}
/////////////////////////////
/// Build inner codebook
/////////////////////////////



void printresults(int n, int point, double pinsdel, int numSeq, int num_blocks, int thread)
{
    int i;
    FILE* rescom;
    char linia[500];
    FILE* rescomB;
    char liniaB[500];

    if (output_type == 0)
    {
        //sprintf(liniaB, "AIR_DNAChannel_WM_MSeq_NumSeq%d_Orate_LLR.txt", numSeq);
        sprintf(liniaB, "Histogram_WM_DNAChannel_p%f.txt", pinsdel);

        // printf("%d %f %e %f\n", n, pinsdel, DTboundAv[max_s][point] / num_blocks, Inf_density[point] * 0.5 / num_blocks);


         //if (point == 0)
         //{
        rescomB = fopen(liniaB, "w");
        //fprintf(rescomB, "pid DT\n");
        //fprintf(rescomB, "block ID\n");
    //}
    //else
        //rescomB = fopen(liniaB, "a+");

    //fprintf(rescomB, "%f %e %f\n", pinsdel, DTboundAv[max_s][point] / num_blocks, Inf_density[point] * 0.5 / num_blocks);
        for (i = 0; i < num_blocks - 1; i++)
        {
            fprintf(rescomB, "%d %e\n", i, Inf_density[i] * 0.5);
        }
        fclose(rescomB);
    }
    else
    {
        sprintf(liniaB, "Output parallel/DT_bound_(%d,%d)TVC_DNA_NumSeq%d_N%d/thread_%d.txt", Def_nWM, Def_kWM, numSeq, Def_nInner, thread);
        //sprintf(liniaB, "Output parallel/DT_bound_DNA_NumSeq%d_N%d/thread_%d.txt", numSeq, Def_nInner, thread);
        //printf("%f %e %f\n", pinsdel, DTboundAv[max_s][point]/ num_blocks, Inf_density[point] * 0.5 / num_blocks);


        if (point == 0)
        {
            rescomB = fopen(liniaB, "w");
            //fprintf(rescomB, "pid DT\n");
        }
        else
            rescomB = fopen(liniaB, "a+");

        /*
        for (i = 0; i < num_blocks - 1; i++)
        {
            fprintf(rescomB, "%d %e\n", thread * 30 + i, Inf_density[i] * 0.5);
        }
        fclose(rescomB);
        */

        //fprintf(rescomB, "%f %e %d\n", pinsdel, DTboundAv[max_s][point] / num_blocks, num_blocks);

        if (bound_type == 1)
            for (i = 0; i < RCUs_intervalpoints; i++)
                fprintf(rescomB, "%f %e %d %f ", pinsdel, DTboundAv[i][point] - pow(2, -RCUs_intervaldelta * (i + 1) * n), numb_blocks[i], RCUs_intervaldelta * (i + 1));
        else
            for (i = 0; i < RCUs_intervalpoints; i++)
                fprintf(rescomB, "%f %e %d %f ", pinsdel, DTboundAv[i][point], numb_blocks[i], RCUs_intervaldelta * (i + 1));

        fprintf(rescomB, "\n");
        fclose(rescomB);
    }
}




///////////////////////////////////////////////////////
/// Lowest density vector tranlator + watermark encoder
///////////////////////////////////////////////////////
void encode_water(int* codeword, int* transmitted, int n, int nw, int* watermark, int* codebook, int type)
{
    int i, j;
    int ipick, temp;
    int numb = 0;

    int N = n * nw / 2;
    int numb_substrings = added_sequence_numb; //Number of substrings to choose from
    int numb_codebooks = inner_codebook_numb; //Number of inner coodebooks to choose from
    int sparsevect[Def_kInner * Def_nWM] = { 0 };
    int sparsevect_DNA[Def_kInner * Def_nWM / 2] = { 0 };

    if (type == 0)
    {
        for (i = 0; i < N; i++)
        {
            if (unif_ch() >= 0.5)
                watermark[i] = 0 + (unif_ch() >= 0.5);
            else
                watermark[i] = 2 + (unif_ch() >= 0.5);
        }
    }
    else if (type == 1)
    {
        for (i = 0; i < N; i++)
        {
            watermark[i] = 0;
        }
    }
    else if (type == 2)
    {
        for (i = 0; i < N; i++)
        {
            watermark[i] = watermark_strings[unif_int() % numb_substrings][i % nw];
        }
    }
    else if (type == 3)
    {
        for (i = 0; i < N; i++)
        {
            watermark[i] = 0;
            /*
            if (unif_ch() >= 0.5)
                watermark[i] = 0 + (unif_ch() >= 0.5);
            else
                watermark[i] = 2 + (unif_ch() >= 0.5);
            */
        }


        codebook[0] = unif_int() % numb_codebooks;

        for (i = 1; i < n; i++)
        {
            codebook[i] = (codebook[i - 1] + 1 + (unif_int() % (numb_codebooks - 1))) % numb_codebooks;
            //codebook[i] = i % numb_codebooks;
        }

        /*
                       for (i = 0; i < n; i++)
                       {
                           //codebook[i] = (codebook[i-1] + 1 + (unif_int() % (numb_codebooks-1))) % numb_codebooks;
                           codebook[i] = i % numb_codebooks;
                       }
          */
    }

    for (i = 0; i < n; i++)
    {
        if (type != 3)
            numb = LookUpSparse[0][codeword[i]]; //one possible codebook
        else
            numb = LookUpSparse[codebook[i]][codeword[i]]; //n possible codebooks

        for (j = 0; j < nw; j++)
            sparsevect[i * nw + j] = (numb >> j) & 1LL;
    }

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < 2; j++)
        {
            sparsevect_DNA[i] += sparsevect[2 * i + j] << j;
        }
    }

    if (type != 3)
    {
        for (i = 0; i < N; i++)
        {
            transmitted[i] = add_tableDNA[sparsevect_DNA[i]][watermark[i]];
        }
    }
    else
    {
        for (i = 0; i < N; i++)
        {
            //transmitted[i] = sparsevect_DNA[i];
            transmitted[i] = add_tableDNA[sparsevect_DNA[i]][watermark[i]];
        }
    }
}
///////////////////////////////////////////////////////
/// Lowest density vector tranlator + watermark encoder
///////////////////////////////////////////////////////


/////////////////////////
/// Add error vector
/////////////////////////
int add_error_vector(int* transmitted, int* recieved, int n, int nw)
{
    int i;

    int i_trans = 0;
    int drift = 0;
    int N = n * nw / 2;
    int rand;
    i_trans = 0;

    while (true)
    {
        drift = 0;
        i_trans = 0;
        for (i = 0; i_trans < N; i++)
        {
            if (unif_ch() < P_trans)
            {
                //ins = 0;
                if (unif_ch() >= P_subs)
                {
                    recieved[i] = transmitted[i_trans];
                }
                else
                {
                    while (true)
                    {
                        rand = unif_int();
                        if ((rand % 4) != transmitted[i_trans])
                        {
                            recieved[i] = rand % 4;
                            break;
                        }
                        else
                            continue;
                    }
                }
                i_trans++;
            }
            else
            {
                if (unif_ch() > 0.5)
                {
                    recieved[i] = unif_int() % 4;
                    //ins++;
                    drift++;
                }
                else
                {
                    i_trans++;
                    i--;
                    drift--;
                }
            }
        }

        if (abs(drift) <= drift_max)
            break;
        else
            continue;
    }

    return(drift);
}
/////////////////////////
/// Add error vector
/////////////////////////


void generate_information_word(int* datawordBits, int* datawordSymb, int l_frame, int BitsPerSymb)
{
    int i, j;

    int lengthBits = l_frame * BitsPerSymb;

    for (i = 0; i < lengthBits; i++)
    {
        datawordBits[i] = unif_int() % 2;
    }

    for (i = 0; i < l_frame; i++)
    {
        datawordSymb[i] = 0;
        for (j = 0; j < BitsPerSymb; j++)
        {
            datawordSymb[i] += datawordBits[BitsPerSymb * i + j] << j;

        }
    }
}



void encoding(int n, int nWM, int* infwordCCw, int* watermark, int* codeword_water, int* codebook)
{
    //encoding inner convolutional code dealing with insertions and deletions
    //add a watermark sequence
    encode_water(infwordCCw, codeword_water, n, nWM, watermark, codebook, added_sequence_type);

}


double InvJfunction(double MI)
{
    double mIe;

    mIe = pow(-(1. / H1) * log(1. - pow(MI, 1 / H3)) / log(2.), 1. / (2. * H2));

    return(mIe);
}

double Jfunction(double MI)
{
    double mIe;
    double a, b;

    a = pow(MI, 2 * H2);
    b = pow(2, -H1 * a);

    mIe = pow(1 - b, H3);

    return(mIe);
}

void GenerateAprioriInformation(int size, double MI, int* sequence, double extrinsic[][GF])
{
    int i, j, jj;
    int bit[16];
    double in[16];
    double p0[16], p1[16];
    double stnddev, variance, mean;
    double sum;

    if (MI == 1.)
    {
        MI = 0.99999999999999;

        variance = 100.;
        mean = variance / 2;
    }
    else
    {
        stnddev = pow(-(1. / H1) * log(1. - pow(MI, 1 / H3)) / log(2.), 1. / (2. * H2));
        variance = stnddev * stnddev;
        mean = variance / 2;
    }

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < BitsPerSymbol; j++)
        {
            bit[j] = (sequence[i] >> j) & 1;
            in[j] = mean * (2. * bit[j] - 1.) + sqrt(variance) * Gaussian();
            in[j] = exp(in[j]);
            p1[j] = in[j] / (1. + in[j]);
            p0[j] = 1. - p1[j];
        }

        sum = 0.;
        for (jj = 0; jj < GF; jj++)
        {
            extrinsic[i][jj] = 1.;
            for (j = 0; j < BitsPerSymbol; j++)
            {
                if (((jj >> j) & 1) == 1)
                    extrinsic[i][jj] *= p1[j];
                else
                    extrinsic[i][jj] *= p0[j];
            }
            sum += extrinsic[i][jj];
        }

        for (jj = 0; jj < GF; jj++)
        {
            extrinsic[i][jj] /= sum;
        }
    }
}



double Compute_Entropy_Y(int recieved[][2 * Def_nInner], int* drift, int n, int nw, int* watermark, int* codebook, int rep_i, double s)
{
    int i, j, ii, jj, edge, d_i, x_1, x_2;

    int I = Ins_max;

    int mid_point = states_max / 2;
    //int mid_point = 0;
    int n_edge = WatermarkCodeTrellis.nedge;

    int rep = Def_NumSeq;
    int rep_start, rep_tot;
    int count = 0;

    double sum;

    double H_Y = 0;

    int N = n * nw;
    int dr_1[Def_NumSeq], dr_2[Def_NumSeq], length[Def_NumSeq];
    int index[Def_NumSeq], index_watermark, numb;
    int N_drift[Def_NumSeq];
    int string[Def_nWM] = { 0 };
    int string_DNA[Def_nWM / 2] = { 0 };
    double Forward_point;

    int max_length_rec = nw * (Ins_max + 1) + 1;

    int temp_recieved[Def_NumSeq][20];
    int temp_transmitted[Def_nWM / 2];
    int temp_received_int[Def_NumSeq];
    int temp_transmitted_int;

    if (rep_i == -1)
    {
        rep_start = 0;
        rep_tot = rep;
    }
    else
    {
        rep_start = rep_i;
        rep_tot = 1;
    }


    for (i = 0; i < rep; i++)
        N_drift[i] = N + drift[i];

    //Pre-compute branch metrics
    for (jj = rep_start; jj < rep_tot + rep_start; jj++)
    {
        for (d_i = 1; d_i < N_drift[jj] + 1; d_i++)
        {
            for (i = 0; i < pow(4, nw); i++)
            {
                int2vec_q(temp_transmitted, i, nw, 4);

                //printf("d = %d, i = %d\n", d_i, i);

                for (j = 0; j < max_length_rec; j++)
                {
                    if ((d_i - 1) + j < N_drift[jj] + 1)
                    {
                        for (ii = 0; ii < j; ii++)
                            temp_recieved[jj][ii] = recieved[jj][(d_i - 1) + ii];
                    }
                    else
                    {
                        for (ii = 0; ii < N_drift[jj] - (d_i - 1); ii++)
                            temp_recieved[jj][ii] = recieved[jj][(d_i - 1) + ii];
                    }

                    Middle[jj][d_i - 1][i][j] = log(Receiver_metric(temp_recieved[jj], temp_transmitted, j, nw));
                }
            }
        }
    }

    for (i = 0; i < states_max; i++)
        for (j = 0; j < n + 1; j++)
            Forward[i][j] = -INFINITY;

    Forward[mid_point][0] = 0;

    //int representation of transmitted sequence + watermark sequence for all trellis sections
    for (d_i = 1; d_i < n + 1; d_i++)
    {
        for (i = 0; i < GF; i++)
        {
            numb = LookUpSparse[codebook[d_i - 1]][i];

            int2vec_q(string, numb, nw * 2, 2);

            for (j = 0; j < nw; j++)
                string_DNA[j] = 0;

            for (j = 0; j < nw; j++)
            {
                for (ii = 0; ii < 2; ii++)
                {
                    string_DNA[j] += string[2 * j + ii] << ii;
                }
            }

            index_watermark = (d_i - 1) * nw;

            for (ii = 0; ii < nw; ii++)
                temp_transmitted[ii] = add_tableDNA[string_DNA[ii]][watermark[index_watermark + ii]];

            //for (ii = 0; ii < nw; ii++)
                //temp_transmitted[ii] = string[ii] ^ watermark[index_watermark + ii];

            vec2int_q(temp_transmitted, temp_transmitted_int, nw, 4);

            //vec2int_q(temp_transmitted, temp_transmitted_int, nw, GF);

            decide_transm_int[i][d_i - 1] = temp_transmitted_int;
        }
    }

    //Forward recursion
    for (d_i = 1; d_i < n + 1; d_i++)
    {
        for (edge = 0; edge < n_edge; edge++)
        {
            x_1 = WatermarkCodeTrellis.trellis[edge][0];
            for (i = rep_start; i < rep_tot + rep_start; i++)
                dr_1[i] = WatermarkCodeTrellis.trellis[edge][i - rep_start + 1];

            x_2 = WatermarkCodeTrellis.trellis[edge][rep_tot + 1];
            for (i = rep_start; i < rep_tot + rep_start; i++)
                dr_2[i] = WatermarkCodeTrellis.trellis[edge][rep_tot + 1 + i - rep_start + 1];

            count = 0;
            for (i = rep_start; i < rep_tot + rep_start; i++)
            {
                index[i] = (d_i - 1) * nw + dr_1[i];
                length[i] = nw + dr_2[i] - dr_1[i];

                if (index[i] < 0 || index[i] + length[i] > N_drift[i])
                    count++;
            }

            for (i = rep_start; i < rep_tot + rep_start; i++)
                if (dr_2[i] + (n - (d_i)) * Ins_max < N_drift[i] - n * nw)
                    count++;

            if (count > 0)
                continue;

            for (i = rep_start; i < rep_tot + rep_start; i++)
            {
                for (ii = 0; ii < length[i]; ii++)
                    temp_recieved[i][ii] = recieved[i][index[i] + ii];

                vec2int_q(temp_recieved[i], temp_received_int[i], length[i], 4);
            }


            for (i = 0; i < GF; i++)
            {
                temp_transmitted_int = decide_transm_int[i][d_i - 1];

                Forward_point = 0;
                for (ii = rep_start; ii < rep_tot + rep_start; ii++)
                {
                    Forward_point += Middle[ii][index[ii]][temp_transmitted_int][length[ii]];
                }

                Forward[x_2][d_i] = max_starLU(Forward[x_1][d_i - 1] + (Forward_point * s), Forward[x_2][d_i]);
            }//end loop over time t possible drift values
        }//end loop over time t - 1 possible drift values

        sum = 0;

        for (ii = 0; ii < states_max; ii++)
            sum += exp(Forward[ii][d_i]);

        for (ii = 0; ii < states_max; ii++)
            Forward[ii][d_i] -= log(sum);

        H_Y -= log2(sum);
    }//end loop over all transmitted bits

    return(H_Y);
}



double Compute_Entropy_XY(int recieved[][2 * Def_nInner], int* drift, int n, int nw, int* watermark, int* transmitted, int* codebook, int rep_i, double s)
{
    int i, j, ii, jj, edge, d_i, x_1, x_2;

    int I = Ins_max;

    int mid_point = states_max / 2;
    int n_edge = WatermarkCodeTrellis.nedge;

    int rep = Def_NumSeq;
    int rep_start, rep_tot;
    int count = 0;

    double sum;

    double H_XY = 0;

    int max_length_rec = nw * (Ins_max + 1) + 1;

    int N = n * nw;
    int dr_1[Def_NumSeq], dr_2[Def_NumSeq], length[Def_NumSeq];
    int index[Def_NumSeq], index_watermark, numb;
    int N_drift[Def_NumSeq];
    int string[Def_nWM] = { 0 };
    int string_DNA[Def_nWM / 2] = { 0 };
    double Forward_point;


    int temp_recieved[Def_NumSeq][20];
    int temp_transmitted[Def_nWM / 2];
    int temp_received_int[Def_NumSeq];
    int temp_transmitted_int;

    if (rep_i == -1)
    {
        rep_start = 0;
        rep_tot = rep;
    }
    else
    {
        rep_start = rep_i;
        rep_tot = 1;
    }

    for (i = 0; i < states_max; i++)
        for (j = 0; j < n + 1; j++)
            Forward[i][j] = -INFINITY;

    for (i = 0; i < rep; i++)
        N_drift[i] = N + drift[i];

    Forward[mid_point][0] = 0;


    //int representation of transmitted sequence + watermark sequence for all trellis sections
    for (d_i = 1; d_i < n + 1; d_i++)
    {

        index_watermark = (d_i - 1) * nw;

        numb = add_tableDNA[transmitted[index_watermark]][watermark[index_watermark]];

        int2vec_q(string, numb, nw * 2, 2);

        for (j = 0; j < nw; j++)
            string_DNA[j] = 0;

        for (j = 0; j < nw; j++)
        {
            for (ii = 0; ii < 2; ii++)
            {
                string_DNA[j] += string[2 * j + ii] << ii;
            }
        }

        for (ii = 0; ii < nw; ii++)
            temp_transmitted[ii] = transmitted[index_watermark + ii];

        vec2int_q(temp_transmitted, temp_transmitted_int, nw, 4);

        //vec2int_q(temp_transmitted, temp_transmitted_int, nw, GF);

        decide_transm_int[0][d_i - 1] = temp_transmitted_int;
    }

    //Forward recursion
    for (d_i = 1; d_i < n + 1; d_i++)
    {
        index_watermark = (d_i - 1) * nw;

        for (ii = 0; ii < nw; ii++)
            temp_transmitted[ii] = add_tableDNA[transmitted[index_watermark + ii]][watermark[index_watermark + ii]];

        vec2int_q(temp_transmitted, numb, nw, 4);

        for (edge = 0; edge < n_edge; edge++)
        {
            x_1 = WatermarkCodeTrellis.trellis[edge][0];
            for (i = rep_start; i < rep_tot + rep_start; i++)
                dr_1[i] = WatermarkCodeTrellis.trellis[edge][i - rep_start + 1];

            x_2 = WatermarkCodeTrellis.trellis[edge][rep_tot + 1];
            for (i = rep_start; i < rep_tot + rep_start; i++)
                dr_2[i] = WatermarkCodeTrellis.trellis[edge][rep_tot + 1 + i - rep_start + 1];

            count = 0;
            for (i = rep_start; i < rep_tot + rep_start; i++)
            {
                index[i] = (d_i - 1) * nw + dr_1[i];
                length[i] = nw + dr_2[i] - dr_1[i];

                if (index[i] < 0 || index[i] + length[i] > N_drift[i])
                    count++;
            }

            for (i = rep_start; i < rep_tot + rep_start; i++)
                if (dr_2[i] + (n - (d_i)) * Ins_max < N_drift[i] - n * nw)
                    count++;

            if (count > 0)
                continue;

            for (i = rep_start; i < rep_tot + rep_start; i++)
            {
                for (ii = 0; ii < length[i]; ii++)
                    temp_recieved[i][ii] = recieved[i][index[i] + ii];

                vec2int_q(temp_recieved[i], temp_received_int[i], length[i], 4);
            }

            temp_transmitted_int = decide_transm_int[0][d_i - 1];


            Forward_point = 0;
            for (ii = rep_start; ii < rep_tot + rep_start; ii++)
            {
                Forward_point += Middle[ii][index[ii]][temp_transmitted_int][length[ii]];
            }

            Forward[x_2][d_i] = max_starLU(Forward[x_1][d_i - 1] + (Forward_point * s), Forward[x_2][d_i]);
        }//end loop over time t - 1 possible drift values

        sum = 0;

        for (ii = 0; ii < states_max; ii++)
            sum += exp(Forward[ii][d_i]);

        for (ii = 0; ii < states_max; ii++)
            Forward[ii][d_i] -= log(sum);

        H_XY -= log2(sum);
    }//end loop over all transmitted bits

    return(H_XY);
}


void DT_bound(int n, int nWM, int point, double* mIe, int l_interleaver, int numSeq, int* drift, int* watermark, int* infwordCCw, int* transmitted, int* codebook, int block_i, double s)
{
    int j;
    int seq;

    double H_Y = 0, H_XY = 0;

    double Info_density = 0;
    double inner_dim = 0;
    double subs = 0;
    double s_bound;

    bool bound = bound_type;

    if (bound == 0)
        s_bound = s;
    else
        s_bound = 1;

    if (Joint_Decoding == 0)
    {
        //decoding inner code
        for (seq = 0; seq < numSeq; seq++)
        {
            H_Y += Compute_Entropy_Y(receivedword, drift, n, nWM, watermark, codebook, seq, s_bound);
            //H_Y += -logPY(receivedword[seq], n + drift[seq], n) * 1.0 / (log(2));
            H_XY += Compute_Entropy_XY(receivedword, drift, n, nWM, watermark, transmitted, codebook, seq, s_bound);
            //H_XY += -(Receiver_metric_log(receivedword[seq], transmitted, n + drift[seq], n)) * s / (log(2));

        }
    }
    else
    {
        H_Y += Compute_Entropy_Y(receivedword, drift, n, nWM, watermark, codebook, -1, s_bound);
        H_XY += Compute_Entropy_XY(receivedword, drift, n, nWM, watermark, transmitted, codebook, -1, s_bound);
    }


    Info_density = BitsPerSymbol * n + H_Y - H_XY;
    //Inf_density[block_i] = Info_density / (n * BitsPerSymbol);


    if (bound == 0)
    {
        //Compute DT bound
        inner_dim = n / 2 * log2(GF) - 1;

        if (Info_density - inner_dim > 0)
            subs = Info_density - inner_dim;
        else
        {
            subs = 0;
            DTboundis1++;
        }

        *mIe = pow(2, -subs);
    }
    else
    {
        //Compute Converse bound
        inner_dim = n / 2 * log2(GF);

        if (Info_density - inner_dim < -n * (s))
        {
            DTboundis1++;
            subs = 1;
        }
        else
            subs = 0;

        *mIe = subs;
    }
}





int main(int argc, char** argv)
{
    int i;
    int npid;
    int block;
    int s;
    double MInf;

    int num_points;
    int npid_init;
    int num_blocks_max;
    int s_init;
    double s_value;

    int    kInner = Def_kInner;
    int    nInner = Def_nInner;

    int    nWM = Def_nWM;        // Information block length inner code
    int    kWM = Def_kWM;        // Codeword length inner code
    int    nDNAsymb = nInner;

    int    NBitsPerSymbolIn = Def_NBitsPerSymbolIn;

    int    numSeq = Def_NumSeq;
    int    seq;

    int    l_interleaver = kInner; //Length interleaver

    int    codebook[Def_kInner] = { 0 };
    int* infwordWMBit = new int[Def_kInner * Def_NBitsPerSymbolIn];
    int* infwordWMSymb = new int[Def_kInner];
    int* watermark = new int[Def_nInner];
    int* codeword_water = new int[Def_nInner];   //Translated NB-LDPC codeword + watermark vector//Transmitted vector + error vector
    int   drift[Def_NumSeq];                   //# of insertions - # of deletions after transmittion

    bool bound = bound_type;

    for (i = 0; i < kInner; i++) perm_code[i] = i;

    if (output_type == 1)
    {
        l_orig = Def_lstart;
        l_start_orig = Def_lstart;
        vc_un_orig = atoi(argv[1]) * (atoi(argv[7]) + 1);
        vc_unInter_orig = atoi(argv[2]) * (atoi(argv[7]) + 1);
        vc_unInterH_orig = atoi(argv[3]) * (atoi(argv[7]) + 1);
        vc_unch_orig = atoi(argv[4]) * (atoi(argv[7]) + 1);
        MinNumberDTboundis1 = atoi(argv[5]);
        num_points = atoi(argv[6]);
        s_init = atoi(argv[7]);
        //s_init = 30;
    }
    else
    {
        l_orig = Def_lstart;
        l_start_orig = Def_lstart;
        vc_un_orig = Def_vcunstart;
        vc_unInter_orig = Def_vc_unInter;
        vc_unInterH_orig = Def_vc_unInterH;
        vc_unch_orig = 2000;
        num_points = Def_NUM_POINTS;
        MinNumberDTboundis1 = Def_MinNumberDTboundis1;
        s_init = 0;
    }

    initialize_variables();
    /*
    if (bound_type == 0)
        s_points = 2;
    */
    MInf = 0.;

    for (npid = 0; npid < num_points; npid++)
    { // loop on MI points

        Inf_density[npid] = 0;
        max_s = 0;
        num_blocks_max = 1;

        for (s = s_init; s < s_points; s++)
        {
            s_value = RCUs_intervaldelta * (s + 1);
            DTboundis1 = 0;

            numb_blocks[s] = 0;

            initialize_sim(kInner, nWM, npid, nInner, s_value);

            build_inner_codebook(nWM, inner_codebook_design);

            for (block = 0; block < 1000000; block++)
            {
                //Generate information word
                generate_information_word(infwordWMBit, infwordWMSymb, l_interleaver, NBitsPerSymbolIn);

                //Encoding
                encoding(kInner, nWM, infwordWMSymb, watermark, codeword_water, codebook);

                //Channel: insertions and deletions channel
                for (seq = 0; seq < numSeq; seq++)
                {
                    drift[seq] = add_error_vector(codeword_water, receivedword[seq], kInner, nWM);
                }

                //printf("im here\n");
                //compute DT bound
                DT_bound(kInner, nWM / 2, npid, &DTbound[s][npid], l_interleaver, numSeq, drift, watermark, infwordWMSymb, codeword_water, codebook, block, s_value);

                DTboundAv[s][npid] += DTbound[s][npid];

                if (willIstop())
                    break;
            }//END loop on blocks

            DTboundAv[s][npid] = DTboundAv[s][npid] / (block + 1);

            numb_blocks[s] = block + 1;
        }//End loop on s points

        if (output_type == 0)
        {
            //print results for the given simulated point
            printresults(kInner, npid, P_ins, numSeq, num_blocks_max + 1, 0);
        }
        else
            printresults(kInner, npid, P_ins, numSeq, num_blocks_max, atoi(argv[7]));

        //if (DTboundAv[npid] / (block + 1) <= 1e-2 + 0.005 && DTboundAv[npid] / (block + 1) >= 1e-2 - 0.005)
          //  break;
    } // END loop on MI points

    delete[] infwordWMBit;
    delete[] infwordWMSymb;
    delete[] watermark;
    delete[] codeword_water;

    return 1;
}
