///////////////////////////////////////////////////////////////////////////
// Alexandre Graell i Amat
// EXIT Charts LDPC + Convolutional Code for the DNA channel, no it (EXIT curve LDPC code)
// October 2020
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
#define Def_Regular                   0
#define Def_kLDPC                 480//24998   //Information word length outer LDPC
#define Def_nLDPC                 (2*Def_kLDPC)  //Codeword length outer LDPC
#define Def_dv                      3 // Variable node degree
#define Def_dc                      6 // Check node degree

#define Def_NumSeq                   1 //Number of sequences (DNA strand reads)

#define Def_n_conv                  2 // (n,k) inner convolutional code
#define Def_k_conv                  1 // (n,k) inner convolutional code
#define Def_v_conv                  2 // Memory inner convolutional code
#define num_states					4 // Number states convolutional code

#define Def_kInner                    (Def_nLDPC+Def_v_conv) //Information word length inner CC code (including termination bits)
#define Def_nInner                    (2*Def_kInner) //Codeword length inner CC

#define GF						    4 // Galois Feild order

#define bound_type                      1 //0 : DT/RCUs bound
                                          //1 : Converse bound

#define RCUs_interval                  0.15 // Max s vlue for RCUs bound
#define RCUs_intervaldelta             0.01 // Quantization of RCUs interval
#define RCUs_intervalpoints          int(RCUs_interval / RCUs_intervaldelta) + 1 //Number of s points to be computed

//Parameters ID channel
#define Ins_max                      2 //Maxiumum number of consecutive insertions as considered by the decoder
#define Del_max                     1 //Maxiumum number of consecutive insertions

//EXIT Charts definitions
#define Def_NumBlocks           10
#define Def_MIPoints            20

double  H1 = 0.3073;          // Parameters for the computation of J function
double  H2 = 0.8935;          // Parameters for the computation of J function
double  H3 = 1.1064;          // Parameters for the computation of J function
int     BitsPerSymbol;


#define Def_InsDelProbIni         0.19 // Initial Insertion/Deletion probability
#define Def_SubsProb              0.000 // Substitution Probability
#define Def_InsDelProbDelta       0.01 // decrease in bit cross-over probability for each simulation point
#define Def_NUM_POINTS              3 // //Number of points to be simulated

#define output_type					  0 //0 : singular otuput (debugging) 
//1 : output to parralelization file

#define Def_MinNumberDTboundis1      1 // Required number of DT bound is equal to 1

#define inner_decoder_domain		  1 //0: Probability domain
                                        //1: Log domain

#define Def_vc_unInter              2344 // Initial seed generation of an integer (between 0 and a given integer) uniformly at random (used
// when generating the error pattern)
#define Def_vc_unInterH                2000 // Initial seed generation of an integer (between 0 and a given integer) uniformly at random (used
                                         // when generating the parity-check matrix)
#define Def_vcunstart                  21 // Initial seed for uniform distribution (between 0 and 1) NOT used here
#define Def_lstart					   10 // Initial seed for generator of random binary sequences NOT used here
#define MAXS_H_INTERVAL			    0.01 //
#define MAXS_H_NUMSAMPLES			1000 //

#define Joint_Decoding              1// 0: Separate decoding of multiple sequences
                                     // 1: Joint decoding of multiple sequences
////////////////////////////////////////
//END Simulation Parameters
////////////////////////////////////////

double jaclog_lookup[MAXS_H_NUMSAMPLES];

int receivedword[Def_NumSeq][2 * Def_nInner * Def_n_conv];

int codeword_ref_CC[Def_nInner] = { 0 };


int perm_code[Def_nInner];


int add_tableDNA[4][4] = { {0, 1, 2, 3 },
                           {1, 0, 3, 2 },
                           {2, 3, 0, 1 },
                           {3, 2, 1, 0 } };

int state_mapper[10000][Def_NumSeq + 1] = { 0 };
int state_trans_conv[num_states][2] = { 0 };
int state_out_conv[num_states][2] = { 0 };
int trellis_term[num_states][2] = { 0 };

vector<vector<vector<int>>> vect2int_lookup(Def_n_conv / 2 * (Ins_max + Del_max) + 1, vector<vector<int>>(Def_n_conv / 2 * (Ins_max + Del_max) + 1, vector<int>(Def_n_conv / 2 * (Ins_max + Del_max) + 1, -1)));


//vector<vector<double>> Forward(164, vector<double>(Def_nk + 1, -1));
//vector<vector<double>> Backward(164, vector<double>(Def_nk + 1, -1));

double Forward[10000][Def_kInner + 1];

//vector<vector<vector<double>>> Middle(1000, vector<vector<double>>(20, vector<double>(Def_nk, -1)));
vector<vector<vector<double>>> Middle(GF, vector<vector<double>>(Def_n_conv / 2 * (Ins_max + Del_max) + 1, vector<double>(pow(2, Def_n_conv* (Ins_max + Del_max)), -1)));


/*************************************************************************/
/**************************   GENREAL VARIABLES   ************************/
/*************************************************************************/


//EXIT Charts Variables
double DTbound[100][2000];
double DTboundAv[100][2000];
double Inf_density[Def_NumBlocks];
int numb_blocks_RCUs[RCUs_intervalpoints] = { 0 };

int    interleaver[Def_nInner + 2];

typedef struct
{
    int nibits;
    int nobits;
    int nstate;
    int ntrans;
    int nedge;
    int trellis[1000000][6 + 2 * Def_NumSeq];

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

int offset;

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

int willIstop()
{
    if (DTboundis1 < MinNumberDTboundis1) return 0;

    return 1;
}


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



/*************************************************************************/
/***************   Initialize variables and simulation   *****************/
/*************************************************************************/
void initialize_variables()
{
    int i;
    int mI;

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

    for (i = 0; i < MAXS_H_NUMSAMPLES; i++)
    {
        jaclog_lookup[i] = log(1 + exp(-i * MAXS_H_INTERVAL));
    }

    l_start = l_start_orig;
    l = l_orig;
    vc_un = vc_un_orig;
    vc_unInter = vc_unInter_orig;
    vc_unInterH = vc_unInterH_orig;
    vc_unch = vc_unch_orig;
    //printf("Number of simulation points %d \n",numero_sim);

    s_points = RCUs_intervalpoints;
}


void finite_state_machine(int u, int& u_1, int& u_2, int& output1, int& output2)
{
    //output1 = u;
    //output2 = u ^ (u_1 ^ u_2) ^ u_2;
    output1 = u ^ u_2;
    output2 = u ^ u_1 ^ u_2;

    //u_temp = u_2;
    u_2 = u_1;
    //u_1 = u_temp ^ u_1 ^ u;
    u_1 = u;
}

/*
void finite_state_machine(int u, int& u_1, int& u_2, int& u_3, int& u_4, int& u_5, int& u_6, int& output1, int& output2, int& output3, int& output4, int& output5, int& output6)
{
    output1 = u ^ u_1 ^ u_2 ^ u_3 ^ u_5 ^ u_6;
    output2 = u ^ u_1 ^ u_3 ^ u_6;
    output3 = u ^ u_2 ^ u_3 ^ u_4 ^ u_6;
    output4 = u ^ u_2 ^ u_3 ^ u_4 ^ u_6;
    output5 = u ^ u_1 ^ u_2 ^ u_5 ^ u_6;
    output6 = u ^ u_2 ^ u_3 ^ u_4 ^ u_5 ^ u_6;

    u_6 = u_5;
    u_5 = u_4;
    u_4 = u_3;
    u_3 = u_2;
    u_2 = u_1;
    u_1 = u;
}
*/

/*
void finite_state_machine(int u, int& u_1, int& u_2, int& u_3, int& u_4, int& output1, int& output2)
{
    int u_temp;

    output1 = u ^ u_1 ^ u_3;
    output2 = u ^ u_2 ^ u_4;

    //u_temp = u ^ u_3 ^ u_4;

    u_4 = u_3;
    u_3 = u_2;
    u_2 = u_1;
    u_1 = u;
}
*/

/*
void finite_state_machine(int u, int& u_1, int& u_2, int& u_3, int& u_4, int& u_5, int& u_6, int& output1, int& output2)
{
    int u_temp;

    output1 = u ^ u_1 ^ u_2 ^ u_3 ^ u_6;
    output2 = u ^ u_2 ^ u_3 ^ u_5 ^ u_6;

    //u_temp = u ^ u_3 ^ u_4;

    u_6 = u_5;
    u_5 = u_4;
    u_4 = u_3;
    u_3 = u_2;
    u_2 = u_1;
    u_1 = u;
}
*/

void buildtrellis_CCwB(int n_conv)
{
    int i, j, k;
    int numb;
    int istateCC;
    int fstateCC;
    int ibit;
    int count = 0;
    int rep = Def_NumSeq; //Number of sequences to decode

    int state[Def_v_conv] = { 0 }; //state encoder
    int obits[Def_n_conv] = { 0 }; //output bits

    bool cond[Def_NumSeq] = { 0 };

    WatermarkCodeTrellis.nibits = Def_k_conv;
    WatermarkCodeTrellis.nobits = Def_n_conv;
    WatermarkCodeTrellis.nstate = num_states * pow(drift_max + 1, rep);

    //state CC
    for (i = 0; i < num_states; i++)
    {
        for (int ibit = 0; ibit <= 1; ibit++)
        {
            for (j = 0; j < Def_v_conv; j++)
            {
                state[j] = ((i >> j) & 1);
            }

            //finite_state_machine(ibit, state[0], state[1], state[2], state[3], state[4], state[5], obits[0], obits[1]);
            //finite_state_machine(ibit, state[0], state[1], state[2], state[3], obits[0], obits[1]);
            finite_state_machine(ibit, state[0], state[1], obits[0], obits[1]);

            numb = 0;
            for (j = 0; j < n_conv; j++)
                numb += obits[j] << j;

            state_out_conv[i][ibit] = numb;

            numb = 0;
            for (j = 0; j < Def_v_conv; j++)
                numb += state[j] << j;

            state_trans_conv[i][ibit] = numb;
        }
    }
    //state CC

    for (int x_2 = 0; x_2 < WatermarkCodeTrellis.nstate; x_2++)
    {
        //count = 0;
        for (int x_1 = 0; x_1 < WatermarkCodeTrellis.nstate; x_1++)
        {
            int sum = 0;

            for (i = 0; i < rep; i++)
            {
                if (state_mapper[x_2][i + 1] - state_mapper[x_1][i + 1] > Ins_max || state_mapper[x_1][i + 1] - state_mapper[x_2][i + 1] > Del_max)
                {
                    cond[i] = 1;
                    sum += 1;
                }
            }

            if (sum > 0 || (state_mapper[x_2][0] != state_trans_conv[state_mapper[x_1][0]][0] && state_mapper[x_2][0] != state_trans_conv[state_mapper[x_1][0]][1]))
                continue;
            else
            {
                if (state_mapper[x_2][0] == state_trans_conv[state_mapper[x_1][0]][0])
                {
                    WatermarkCodeTrellis.trellis[count][0] = x_1;
                    WatermarkCodeTrellis.trellis[count][1] = state_mapper[x_1][0];
                    for (i = 0; i < rep; i++)
                        WatermarkCodeTrellis.trellis[count][2 + i] = state_mapper[x_1][i + 1];
                    WatermarkCodeTrellis.trellis[count][3 + rep - 1] = 0;
                    WatermarkCodeTrellis.trellis[count][4 + rep - 1] = state_out_conv[state_mapper[x_1][0]][0];
                    WatermarkCodeTrellis.trellis[count][5 + rep - 1] = x_2;
                    WatermarkCodeTrellis.trellis[count][6 + rep - 1] = state_mapper[x_2][0];
                    for (i = 0; i < rep; i++)
                        WatermarkCodeTrellis.trellis[count][7 + rep - 1 + i] = state_mapper[x_2][i + 1];

                    count++;
                }
                else
                {
                    WatermarkCodeTrellis.trellis[count][0] = x_1;
                    WatermarkCodeTrellis.trellis[count][1] = state_mapper[x_1][0];
                    for (i = 0; i < rep; i++)
                        WatermarkCodeTrellis.trellis[count][2 + i] = state_mapper[x_1][i + 1];
                    WatermarkCodeTrellis.trellis[count][3 + rep - 1] = 1;
                    WatermarkCodeTrellis.trellis[count][4 + rep - 1] = state_out_conv[state_mapper[x_1][0]][1];
                    WatermarkCodeTrellis.trellis[count][5 + rep - 1] = x_2;
                    WatermarkCodeTrellis.trellis[count][6 + rep - 1] = state_mapper[x_2][0];
                    for (i = 0; i < rep; i++)
                        WatermarkCodeTrellis.trellis[count][7 + rep - 1 + i] = state_mapper[x_2][i + 1];

                    count++;
                }
            }
        }
    }

    WatermarkCodeTrellis.nedge = count;


    //trellis termination
    for (i = 0; i < num_states; i++)
    {
        int u[Def_v_conv] = { 0 };
        int u_temp[Def_v_conv] = { 0 };
        int u_in[Def_n_conv] = { 0 };
        int temp[Def_n_conv] = { 0 };

        for (j = 0; j < Def_v_conv; j++)
        {
            u[j] = (i >> j) & 1LL;
        }



        for (j = 0; j < num_states; j++)
        {
            for (k = 0; k < Def_v_conv; k++)
                u_temp[k] = u[k];

            for (k = 0; k < Def_n_conv; k++)
                temp[k] = 0;

            for (k = 0; k < Def_n_conv; k++)
            {
                u_in[k] = (k >> k) & 1LL;
            }

            for (k = 0; k < Def_n_conv; k++)
            {
                //finite_state_machine(u_in[k], u_temp[0], u_temp[1], u_temp[2], u_temp[3], u_temp[4], u_temp[5], temp[0], temp[1]);
                //finite_state_machine(u_in[k], u_temp[0], u_temp[1], u_temp[2], u_temp[3], temp[0], temp[1]);
                finite_state_machine(u_in[k], u_temp[0], u_temp[1], temp[0], temp[1]);
            }

            if (u_temp[0] == 0 && u_temp[1] == 0)
            {
                trellis_term[i][0] = u_in[0];
                trellis_term[i][1] = u_in[1];
                break;
            }
        }
    }



}


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


/////////////////////////////
/// Calculate receiver metric
/////////////////////////////
double Receiver_metric(int* temp_recieved, int* temp_trans, int length1, int length2)
{
    int i, j;
    double F_pq[10][10];

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


////////////////////////////////////////
/// Pre-compute branch/receiver metrics
////////////////////////////////////////
void compute_BranchMetrics(int n_conv, int q)
{
    int i, j, ii, jj;
    int max_lengthY = n_conv * (Ins_max + Del_max);

    int* x = new int[n_conv]; //transmitted symbol
    int* y = new int[max_lengthY]; //Received sequence

    int y_int; //decimal value of received sequence

    for (i = 0; i < pow(q, n_conv); i++)
    {//loop over all possible transmitted symbols X
        x[0] = i;

        for (j = 0; j <= max_lengthY; j++)
        {//loop over all possible lengths of received sequence Y
            for (ii = 0; ii < pow(q, j); ii++)
            {
                int2vec_q(y, ii, max_lengthY, q); //convert int reprsentation of Y to a vect representation

                vec2int_q(y, y_int, max_lengthY, q);

                vect2int_lookup[y[0]][y[1]][y[2]] = y_int;

                if (inner_decoder_domain == 0)
                    Middle[i][j][ii] = pow(Receiver_metric(y, x, j, n_conv), 1);
                else
                    Middle[i][j][ii] = log(pow(Receiver_metric(y, x, j, n_conv), 1));
            }
        }
    }

    delete[] x;
    delete[] y;
}
////////////////////////////////////////
/// Pre-compute branch/receiver metrics
////////////////////////////////////////

void initialize_sim(int n, int n_conv, int sim)
{
    int i, j;

    l_start = l_start_orig;
    l = l_orig;
    vc_un = vc_un_orig;
    vc_unInter = vc_unInter_orig;
    vc_unInterH = vc_unInterH_orig;
    vc_unch = vc_unch_orig;

    P_ins = Def_InsDelProbIni - sim * Def_InsDelProbDelta;
    P_del = Def_InsDelProbIni - sim * Def_InsDelProbDelta;
    //P_del = 0;
    P_trans = 1 - P_ins - P_del;
    P_subs = Def_SubsProb;

    int B = 5 * sqrt((n / 2) * P_ins / (1 - P_ins));
    int rep = Def_NumSeq;

    drift_max = 2 * B;
    states_max = num_states * (drift_max + 1);

    for (i = 0; i < num_states * pow(drift_max + 1, rep); i++)
    {
        state_mapper[i][0] = i / pow(drift_max + 1, rep);
        for (j = 0; j < rep - 1; j++)
        {
            state_mapper[i][j + 1] = -B + ((i % int(pow(drift_max + 1, rep - j))) / ((rep - (j + 1)) * drift_max + 1));
        }
        state_mapper[i][rep] = -B + i % (drift_max + 1);
    }

    buildtrellis_CCwB(n_conv);

    //Pre-compute branch/receiver metrics P(Y|X) for all X and Y
    compute_BranchMetrics(n_conv / 2, GF);

    for (i = 0; i < Def_nLDPC; i++) perm_code[i] = i;
}






void printresults(int point, double pinsdel, int numSeq, int num_blocks, int thread)
{
    int i;
    FILE* rescom;
    char linia[500];
    FILE* rescomB;
    char liniaB[500];

    if (output_type == 0)
    {
        //sprintf(liniaB, "AIR_DNAChannel_WM_MSeq_NumSeq%d_Orate_LLR.txt", numSeq);
        sprintf(liniaB, "Histogram_CC_DNAChannel_p%f.txt", pinsdel);

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
        sprintf(liniaB, "Output parallel/DT_bound_CC_DNA_NumSeq%d_N%d/thread_%d.txt", numSeq, Def_kInner, thread);

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
        if (bound_type == 0)
            for (i = 0; i < RCUs_intervalpoints; i++)
                fprintf(rescomB, "%f %e %d %f ", pinsdel, DTboundAv[i][point] - pow(2, -i * Def_kInner), numb_blocks_RCUs[i], RCUs_intervaldelta * (i + 1));
        else
            for (i = 0; i < RCUs_intervalpoints; i++)
                fprintf(rescomB, "%f %e %d %f ", pinsdel, DTboundAv[i][point], numb_blocks_RCUs[i], RCUs_intervaldelta * (i + 1));
        
        fprintf(rescomB, "\n");
        fclose(rescomB);
    }
}



/////////////////////////
/// Encoder CC Watermark
/////////////////////////
void encode_CC(int* LDPC, int* CC, int n, int nk)
{
    int i;
    int u_1 = 0;
    int u_2 = 0;
    int u_3 = 0;
    int u_4 = 0;
    int u_5 = 0;
    int u_6 = 0;

    int numb = 0;

    for (i = 0; i < n; i++)
        codeword_ref_CC[i] = 0;

    for (i = 0; i < nk - Def_v_conv; i++)
    {
        //finite_state_machine(LDPC[i], u_1, u_2, u_3, u_4, u_5, u_6, CC[Def_n_conv * i], CC[Def_n_conv * i + 1]);
        //finite_state_machine(LDPC[i], u_1, u_2, u_3, u_4, CC[Def_n_conv * i], CC[Def_n_conv * i + 1]);
        finite_state_machine(LDPC[i], u_1, u_2, CC[Def_n_conv * i], CC[Def_n_conv * i + 1]);
    }

    //numb += u_1 + u_2 * 2 + u_3 * 4 + u_4 * 8 + u_5 * 16 + u_6 * 32;
    //numb += u_1 + u_2 * 2 + u_3 * 4 + u_4 * 8;
    numb += u_1 + u_2 * 2;

    for (; i < nk; i++)
    {
        //finite_state_machine(0, u_1, u_2, u_3, u_4, CC[Def_n_conv * i], CC[Def_n_conv * i + 1]);
        //finite_state_machine(trellis_term[numb][i - (nk - Def_v_conv)], u_1, u_2, u_3, u_4, u_5, u_6, CC[Def_n_conv * i], CC[Def_n_conv * i + 1]);
        //finite_state_machine(trellis_term[numb][i - (nk - Def_v_conv)], u_1, u_2, u_3, u_4, CC[Def_n_conv * i], CC[Def_n_conv * i + 1]);
        finite_state_machine(trellis_term[numb][i - (nk - Def_v_conv)], u_1, u_2, CC[Def_n_conv * i], CC[Def_n_conv * i + 1]);

        LDPC[i] = trellis_term[numb][i - (nk - Def_v_conv)];
    }
    for (i = 0; i < n; i++)
        codeword_ref_CC[i] = CC[i];
}
/////////////////////////
/// Encoder cc
/////////////////////////


///////////////////////////////////////////////////////
/// Lowest density vector tranlator + wtaermark encoder
///////////////////////////////////////////////////////
void encode_water(int* codeword, int* transmitted, int n, int n_conv, int* watermark)
{
    int i, j;
    int DNA_vect[Def_kInner];

    int nsymb = n / 2;

    for (i = 0; i < nsymb; i++)
    {
        if (unif_ch() >= 0.5)
            watermark[i] = 0 + (unif_ch() >= 0.5);
        else
            watermark[i] = 2 + (unif_ch() >= 0.5);
    }

    int numb;
    for (i = 0; i < nsymb; i++)
    {
        numb = 0;
        for (j = 0; j < n_conv; j++)
        {
            numb += codeword[2 * i + j] << j;
        }

        DNA_vect[i] = numb;
    }

    for (i = 0; i < nsymb; i++)
    {
        transmitted[i] = add_tableDNA[DNA_vect[i]][watermark[i]];
    }

    //    delete[] DNA_vect;
}
///////////////////////////////////////////////////////
/// Lowest density vector tranlator + wtaermark encoder
///////////////////////////////////////////////////////



/////////////////////////
/// Add error vector
/////////////////////////
int add_error_vector(int* transmitted, int* recieved, int n, int nw)
{
    int i;

    int i_trans = 0;
    int drift = 0;
    int ins = 0;
    int N = n;
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
                ins = 0;
                if (unif_ch() >= P_subs)
                {
                    recieved[i] = transmitted[i_trans];
                }
                else
                {
                    while (true)
                    {
                        rand = unif_intH();
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
                    recieved[i] = unif_intH() % 4;
                    ins++;
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

        if (abs(drift) <= drift_max / 2)
            break;
        else
            continue;
    }

    return(drift);
}
/////////////////////////
/// Add error vector
/////////////////////////


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



///////////////////////////////////////
/// P(y)/H(y) calculation in log domain 
///////////////////////////////////////
double Compute_Entropy_Y_LogDomain(int seq, int recieved[][2 * Def_nInner * Def_n_conv], int* drift, int n, int n_conv, int nk, int nkw, int* watermark, double s)
{
    int i, j, ii, jj, edge, d_i, x_1, x_2;

    int mid_point = drift_max;
    int n_edge = WatermarkCodeTrellis.nedge;
    int n_states = WatermarkCodeTrellis.nstate;

    int dr_1[Def_NumSeq], dr_2[Def_NumSeq], length[Def_NumSeq];
    int index[Def_NumSeq], index_watermark;
    int N_drift[Def_NumSeq];

    int numb;
    int count;
    int drift_state = 0;
    int rep = Def_NumSeq;

    int temp_recieved[Def_NumSeq][4];
    int temp_recieved_int[Def_NumSeq] = { 0 };
    int temp_transmitted[Def_n_conv];

    double Forward_point;
    double max_star_point;

    double H_Y = 0;

    for (i = 0; i < rep; i++)
        N_drift[i] = n / 2 + drift[i];

    for (i = 0; i < n_states; i++)
        for (j = 0; j < nk + 1; j++)
            Forward[i][j] = -INFINITY;

    for (i = 0; i < WatermarkCodeTrellis.nstate; i++)
    {
        if (state_mapper[i][0] == 0 && state_mapper[i][1] == 0 && state_mapper[i][2] == 0)
        {
            mid_point = i;
            break;
        }
    }

    Forward[mid_point][0] = 0;

    //Forward recursion
    for (d_i = 1; d_i < nk + 1; d_i++)
    {
        for (edge = 0; edge < n_edge; edge++)
        {
            x_1 = WatermarkCodeTrellis.trellis[edge][0];
            for (i = 0; i < rep; i++)
                dr_1[i] = WatermarkCodeTrellis.trellis[edge][2 + i];
            j = WatermarkCodeTrellis.trellis[edge][3 + rep - 1];
            numb = WatermarkCodeTrellis.trellis[edge][4 + rep - 1];
            x_2 = WatermarkCodeTrellis.trellis[edge][5 + rep - 1];
            for (i = 0; i < rep; i++)
                dr_2[i] = WatermarkCodeTrellis.trellis[edge][7 + rep + i - 1];

            index_watermark = d_i - 1;
            count = 0;
            for (i = 0; i < rep; i++)
            {
                index[i] = (d_i - 1) + dr_1[i];
                length[i] = n_conv / 2 + dr_2[i] - dr_1[i];

                if (index[i] < 0 || index[i] + length[i] > N_drift[i])
                    count++;

                if (d_i > nk - 3000)
                    if (dr_1[i] < (N_drift[i] - nk) - Ins_max * (nk - d_i))
                        count++;
            }

            if (count > 0)
                continue;

            temp_transmitted[0] = add_tableDNA[numb][watermark[index_watermark]];

            for (i = 0; i < rep; i++)
                for (ii = 0; ii < 4; ii++)
                    temp_recieved[i][ii] = 0;

            for (i = 0; i < rep; i++)
            {
                for (ii = 0; ii < length[i]; ii++)
                    temp_recieved[i][ii] = recieved[i][index[i] + ii];

                temp_recieved_int[i] = vect2int_lookup[temp_recieved[i][0]][temp_recieved[i][1]][temp_recieved[i][2]];
                //vec2int_q(temp_recieved, temp_recieved_int,length,GF);
            }

            //Forward recursion calculation
            Forward_point = 0;
            for (i = 0; i < rep; i++)
                Forward_point += Middle[temp_transmitted[0]][length[i]][temp_recieved_int[i]]; //+log(0.5);
            max_star_point = Forward[x_1][d_i - 1] + Forward_point * s;
            Forward[x_2][d_i] = max_star(max_star_point, Forward[x_2][d_i]);
        }//end loop over edges

        //double max = Forward[0][d_i];
        double sum = 0;

        for (ii = 0; ii < n_states; ii++)
            sum += exp(Forward[ii][d_i]);

        for (ii = 0; ii < n_states; ii++)
            Forward[ii][d_i] -= log(sum);

        H_Y -= log2(sum);

    }//end loop over all transmitted bits

    /*
    for (i = 0; i < states_max; i++)
    {
        //if (Forward[i][nk] != -INFINITY)
            H_Y = max_starLU(Forward[i][nk], H_Y);
    }

    H_Y = log2(exp(H_Y));
    */
    return(H_Y / nk);
}
///////////////////////////////////////
/// P(y)/H(y) calculation in log domain 
///////////////////////////////////////



////////////////////////////////////////////
/// P(x,y)/H(x,y) calculation in log domain  
////////////////////////////////////////////
double Compute_Entropy_XY_LogDomain(int seq, int recieved[][2 * Def_nInner * Def_n_conv], int* transmitted, int* drift, int n, int n_conv, int nk, int nkw, int* watermark, double s)
{
    int i, j, ii, jj, edge, d_i, x_1, x_2;

    int mid_point = drift_max;
    int n_edge = WatermarkCodeTrellis.nedge;
    int n_states = WatermarkCodeTrellis.nstate;

    int dr_1[Def_NumSeq], dr_2[Def_NumSeq], length[Def_NumSeq];
    int index[Def_NumSeq], index_watermark;
    int N_drift[Def_NumSeq];

    int numb;
    int count;
    int drift_state = 0;
    int rep = Def_NumSeq;

    int temp_recieved[Def_NumSeq][4];
    int temp_recieved_int[Def_NumSeq] = { 0 };
    int temp_transmitted[Def_n_conv];

    double Forward_point;
    double max_star_point;

    double H_XY = 0;
    for (i = 0; i < rep; i++)
        N_drift[i] = n / 2 + drift[i];

    for (i = 0; i < n_states; i++)
        for (j = 0; j < nk + 1; j++)
            Forward[i][j] = -INFINITY;

    for (i = 0; i < WatermarkCodeTrellis.nstate; i++)
    {
        if (state_mapper[i][0] == 0 && state_mapper[i][1] == 0 && state_mapper[i][2] == 0)
        {
            mid_point = i;
            break;
        }
    }

    Forward[mid_point][0] = 0;

    //Forward recursion
    for (d_i = 1; d_i < nk + 1; d_i++)
    {

        for (edge = 0; edge < n_edge; edge++)
        {
            x_1 = WatermarkCodeTrellis.trellis[edge][0];
            for (i = 0; i < rep; i++)
                dr_1[i] = WatermarkCodeTrellis.trellis[edge][2 + i];
            j = WatermarkCodeTrellis.trellis[edge][3 + rep - 1];
            numb = WatermarkCodeTrellis.trellis[edge][4 + rep - 1];
            x_2 = WatermarkCodeTrellis.trellis[edge][5 + rep - 1];
            for (i = 0; i < rep; i++)
                dr_2[i] = WatermarkCodeTrellis.trellis[edge][7 + rep + i - 1];

            index_watermark = d_i - 1;
            count = 0;
            for (i = 0; i < rep; i++)
            {
                index[i] = (d_i - 1) + dr_1[i];
                length[i] = n_conv / 2 + dr_2[i] - dr_1[i];

                if (index[i] < 0 || index[i] + length[i] > N_drift[i] || numb != add_tableDNA[transmitted[index_watermark]][watermark[index_watermark]])
                    count++;

                if (d_i > nk - 3000)
                    if (dr_1[i] < (N_drift[i] - nk) - Ins_max * (nk - d_i))
                        count++;
            }

            if (count > 0)
                continue;

            temp_transmitted[0] = transmitted[index_watermark];

            for (i = 0; i < rep; i++)
                for (ii = 0; ii < 4; ii++)
                    temp_recieved[i][ii] = 0;

            for (i = 0; i < rep; i++)
            {
                for (ii = 0; ii < length[i]; ii++)
                    temp_recieved[i][ii] = recieved[i][index[i] + ii];

                temp_recieved_int[i] = vect2int_lookup[temp_recieved[i][0]][temp_recieved[i][1]][temp_recieved[i][2]];
                //vec2int_q(temp_recieved, temp_recieved_int,length,GF);
            }

            //Forward recursion calculation
            Forward_point = 0;
            for (i = 0; i < rep; i++)
                Forward_point += Middle[temp_transmitted[0]][length[i]][temp_recieved_int[i]]; //+log(0.5);
            max_star_point = Forward[x_1][d_i - 1] + Forward_point * s;
            Forward[x_2][d_i] = max_star(max_star_point, Forward[x_2][d_i]);
        }//end loop over edges

        double sum = 0;

        for (ii = 0; ii < n_states; ii++)
            sum += exp(Forward[ii][d_i]);

        for (ii = 0; ii < n_states; ii++)
            Forward[ii][d_i] -= log(sum);

        H_XY -= log2(sum);

    }//end loop over all transmitted bits

    /*
    for (i = 0; i < states_max; i++)
    {
        //if (Forward[i][nk] != -INFINITY)
        H_XY = max_starLU(Forward[i][nk], H_XY);
    }

    H_XY = log2(exp(H_XY));
    */
    return(H_XY / nk);
}
////////////////////////////////////////////
/// P(x,y)/H(x,y) calculation in log domain  
////////////////////////////////////////////


void generate_information_word(int* datawordf, int l_frame)
{
    int i;

    for (i = 0; i < l_frame; i++)
    {
        datawordf[i] = unif_int() % 2;
    }
}

void encodeOuterCC(int* inputWord, int* outputWord, int kCCf, int nuCCf)
{
    int i, u;
    int xone, xtwo;
    int state[2] = { 0,0 };
    int FinalState;

    for (i = 0; i < kCCf; i++)
    {
        u = inputWord[i];
        xone = u ^ state[1];
        xtwo = u ^ state[0] ^ state[1];
        state[1] = state[0];
        state[0] = u;

        outputWord[2 * i] = xone;
        outputWord[2 * i + 1] = xtwo;

    }

    for (; i < kCCf + nuCCf; i++)
    {
        u = 0;
        xone = u ^ state[1];
        xtwo = u ^ state[0] ^ state[1];
        state[1] = state[0];
        state[0] = u;

        outputWord[2 * i] = xone;
        outputWord[2 * i + 1] = xtwo;
    }

    FinalState = state[0] + (state[1] << 1);

    if (FinalState != 0)
    {
        printf("argggg!");
        exit(1);
    }
}


////////////////////////////////
//// Generate Interleaver
////////////////////////////////

void generateInterleaver(int l_interleaver)
{
    int i;
    int ipick, temp;
    for (i = 0; i < l_interleaver; i++)
    {
        ipick = i + unif_int() % (l_interleaver - i);
        temp = perm_code[i];
        perm_code[i] = perm_code[ipick];
        perm_code[ipick] = temp;
    }

    for (i = 0; i < l_interleaver; i++)
        interleaver[i] = perm_code[i];

}





void encoding(int kLDPC, int nLDPC, int n, int n_conv, int* dataword, int* codewordLDPC, int* infwordCCw, int* codeword_CC, int* watermark, int* codeword_water)
{
    //encoding inner convolutional code dealing with insertions and deletions
    encode_CC(infwordCCw, codeword_CC, n, nLDPC + Def_v_conv);
    //add a watermark sequence
    encode_water(codeword_CC, codeword_water, n, n_conv, watermark);

}

/////////////////////////
/// Parallel Decoder
/////////////////////////



/////////////////////////
/// Parallel Decoder
/////////////////////////


void DT_bound(int kinner, int ninner, int dv, int dc, int n, int n_conv, int nk, int point, double* mIe, int l_interleaver, int* codewordLDPC, int numSeq, int* drift, int* watermark, int* infwordCCw, int* transmitted, int block_i, double s)
{
    int j;
    int seq;

    double H_Y = 0;
    double H_XY = 0;

    double Info_density = 0;
    double inner_dim = 0;
    double subs = 0;
    double s_bound;

    bool bound = bound_type;

    if (bound == 0)
        s_bound = s;
    else
        s_bound = 1;


    //decoding inner code
    if (Joint_Decoding == 0)
    {
        for (seq = 0; seq < numSeq; seq++)
        {
            H_Y += Compute_Entropy_Y_LogDomain(seq, receivedword, drift, n, n_conv, nk, n_conv, watermark, s_bound);
            H_XY += Compute_Entropy_XY_LogDomain(seq, receivedword, transmitted, drift, n, n_conv, nk, n_conv, watermark, s_bound);
        }
    }
    else
    {
        H_Y += Compute_Entropy_Y_LogDomain(-1, receivedword, drift, n, n_conv, nk, n_conv, watermark, s_bound);
        H_XY += Compute_Entropy_XY_LogDomain(-1, receivedword, transmitted, drift, n, n_conv, nk, n_conv, watermark, s_bound);
    }


    Info_density =  (1 + H_Y - H_XY) * nk;
    //Inf_density[block_i] = Info_density / (n * BitsPerSymbol);


    if (bound == 0)
    {
        //Compute DT bound
        inner_dim = nk / 2 * log2(2) - 1;

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
        inner_dim = nk / 2 * log2(2);

        if (Info_density - inner_dim <= -nk * (s))
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
    int i, s;
    int npid;
    int block;
    double MInf;

    int num_points;
    int npid_init;
    int numb_blocks;
    int num_blocks_max;
    int s_init;
    double s_value;

    int    kLDPC = Def_kLDPC;       // Information block length outer CC
    int    nLDPC = kLDPC * 2; // Codeword length outer CC
    int    dv = Def_dv;       // Variable node degree
    int    dc = Def_dc;       // Check node degree

    int    nk = Def_kInner;      // Information block length inner code
    int    n = Def_nInner;        // Codeword length inner code
    int    nDNAsymb = n / 2;

    int    numSeq = Def_NumSeq;
    int    seq;

    int    l_interleaver = nLDPC; //Length interleaver

    int    n_conv = Def_n_conv;

    int    dataword[Def_kLDPC];
    int    codewordLDPC[Def_nLDPC];
    int    watermark[Def_nInner];
    int    codeword_CC[Def_nInner];         //CC codeword
    int    codeword_water[Def_nInner];   //Translated NB-LDPC codeword + watermark vector//Transmitted vector + error vector
    int    codeworddec_LDPC[Def_nLDPC];
    int    drift[Def_NumSeq];                   //# of insertions - # of deletions after transmittion
    int rep = Def_NumSeq;


    //CC parameters
    int infwordCCw[Def_kInner];

    for (i = 0; i < nLDPC; i++) perm_code[i] = i;

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
        //s_init = atoi(argv[7]);
        s_init = 0;
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

    MInf = 0.;

    for (npid = 0; npid < num_points; npid++)
    { // loop on MI points

        Inf_density[npid] = 0;
        max_s = 0;
        num_blocks_max = 1;

        for (s = s_init; s < s_points; s++)
        {
            s_value = RCUs_intervaldelta * (s + 1);

            numb_blocks_RCUs[s] = 0;

            initialize_sim(n, n_conv, npid);

            DTboundis1 = 0;

            for (block = 0; block < 100000000; block++)
            {
                //Generate information word
                generate_information_word(infwordCCw, l_interleaver);

                //Encoding
                encoding(kLDPC, nLDPC, n, n_conv, dataword, codewordLDPC, infwordCCw, codeword_CC, watermark, codeword_water);

                //Channel: insertions and deletions channel
                for (seq = 0; seq < numSeq; seq++)
                {
                    drift[seq] = add_error_vector(codeword_water, receivedword[seq], nDNAsymb, n_conv);
                }

                DT_bound(kLDPC, nLDPC, dv, dc, n, n_conv, nk, npid, &DTbound[s][npid], l_interleaver, codewordLDPC, numSeq, drift, watermark, infwordCCw, codeword_water, block, s_value);

                DTboundAv[s][npid] += DTbound[s][npid];

                if (willIstop())
                    break;

            }//END loop on blocks
            
            DTboundAv[s][npid] = DTboundAv[s][npid] / (block + 1);

            numb_blocks_RCUs[s] = block + 1;
        }

        if (output_type == 0)
        {
            //print results for the given simulated point
            printresults(npid, P_ins, numSeq, num_blocks_max + 1, 0);
        }
        else
            printresults(npid, P_ins, numSeq, num_blocks_max, atoi(argv[7]));

        //if (DTboundAv[max_s][npid] / (block + 1) <= 1e-2 + 0.005 && DTboundAv[max_s][npid] / (block + 1) >= 1e-2 - 0.005)
            //break;
    } // END loop on MI points
    return 1;
}
