///////////////////////////////////////////////////////////////////////////
// Issam Maarouf                                         
// Forward-Backward Algorithm implementation for a Hidden Markov Model (HMM)
// Jan 2019
///////////////////////////////////////////////////////////////////////////

//Include

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <limits.h>
//#include <complex.h>
//#include <bits/stdc++.h> 
#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <fstream>

using namespace std;

//Define

#define Ins_max                     2 //Maxiumum number of consecutive insertions

#define Def_n                     240 // Codeword length
#define Def_nk                    120 // Number of check nodes
#define Def_nw                      8 // Codeword length
#define Def_nkw                     4 // Number of check nodes
#define Def_dv                      4 // Variable node degree
#define Def_dc                      5 // Check node degree
#define Def_Bk                      3 // Number of rows of the protograph matrix
#define Def_Bn                      6 // Number of columns of the protograph matrix
#define GF						    16 // Galius Feild order 
#define f					   0.3125 //Mean density of sparse vectors

#define Def_InsDelProbIni        0.12// Initial Insertion/Deletion probability
#define Def_SubsProb             0.00// Substitution Probability
#define Def_InsDelProbDelta      0.01// decrease in bit cross-over probability for each simulation point
#define Def_NUM_POINTS             4// //Number of points to be simulated

#define Def_MaxNumIt                  100 // Maximum number of iteration of the iterative decoding
#define Def_MinNumberFramesError      100 // Required number of frames in error to validate a simulation point
#define Def_CheckFrames               100// It prints intermediate results on the screen every Def_CheckFrames frames

#define LDPC_build_type				  0 //0 : bulding a random LDPC code 
										//1 : reading LDPC code from external file
										//2: Construct LDPC code from given distribution using PEG algorithm
					//3 : Construct LDPC code from degree distribution using PEG algorithm

#define Def_dv_irr					  6 //maximum VN degree of irregular LDPC code
#define Def_dc_irr                    7 //maximum CN degree of irregular LDPC code

#define output_type					  0 //0 : singular otuput (debugging) 
										//1 : output to parralelization file

#define inner_decoder_domain		  0 //0: Probability domain
										//1: Log domain

//Inner codebook choice
#define inner_codebook_design         0 //0 : use the Davey-MacKay construction
										//1 : use construction that minimizes the change probability (CP)
										//2 : manually input the codebook
										//3 : manually input n number of codebooks
										//4 : VT code
										//5 : use a random codebook


#define a_VT                          1 // a parameter of VT code.

#define inner_codebook_numb			  4 // number of inner codebooks

#define added_sequence_type			  0 //0 : use a random sequnce 
										//1 : no random sequence
										//2 : use n best substrings
										//3 : choose between n codeboos

#define added_sequence_numb			 4 //Number of substrings you can choose to add to the transmitted data

#define Def_vc_unInter              2344 // Initial seed generation of an integer (between 0 and a given integer) uniformly at random (used
										 // when generating the error pattern)
#define Def_vc_unInterH                2000 // Initial seed generation of an integer (between 0 and a given integer) uniformly at random (used
										 // when generating the parity-check matrix)
#define Def_vcunstart                  1 // Initial seed for uniform distribution (between 0 and 1) NOT used here
#define Def_lstart					   1 // Initial seed for generator of random binary sequences NOT used here
#define MAXS_H_INTERVAL			    0.01 //
#define MAXS_H_NUMSAMPLES			1000 //
#define Rep_Factor					   1 // Number of times a symbole will be repeated(also number of parallel channels)

#define Joint_Decoding              0// 0: Separate decoding of multiple sequences
									 // 1: Joint decoding of multiple sequences 

#define Turbo_Iterations			   100 //Number of iterations between WM and LDPC decoders
#define min(x,y) ((x) < (y) ? (x) : (y)) //calculate minimum between two values


int Bmatrix[Def_Bk][Def_Bn] = { {1,1,0,0,0,3},
								{0,1,1,2,1,0},
								{1,1,1,0,1,1} };     //protograph matrix

/*
int Bmatrix[Def_Bk][Def_Bn] = { {1,2,2,2},
								{1,0,0,2} };     //protograph matrix
*/
//int Bmatrix[Def_Bk][Def_Bn] = { 3,3 };     //protograph matrix

//WiMax code
/*
double degree_dist_VN[Def_dv_irr + 1] = { 0.0,0.0,0.289474,0.31579,0.0,0.0,0.394737};
double degree_dist_CN[Def_dc_irr + 1] = { 0.0,0.0,0.0,0.0,0.0,0.0,0.6315792963989104,0.36842070360108964};
*/

double degree_dist_VN[Def_dv_irr + 1] = { 0.0,0.0,0.28947,0.3158,0.0,0.0,0.3947 };
double degree_dist_CN[Def_dc_irr + 1] = { 0.0,0.0,0.0,0.0,0.0,0.0,0.6316,0.3684 };


/*
double degree_dist_VN[Def_dv_irr + 1] = {0.0,0.0,0.254924,0.304387,0.0,0.0,0.0,0.0,0.100373,0.050626,0.28969};
double degree_dist_CN[Def_dc_irr + 1] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.880597,0.071173,0.0,0.0,0.0,0.0,0.0,0.022427,0.025803};
*/

/*
double degree_dist_VN[Def_dv_irr + 1] = { 0.0,0.0,0.282575,0.361403,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.356022};
double degree_dist_CN[Def_dc_irr + 1] = { 0.0,0.0,0.0,0.167580,0.0,0.0,0.068519,0.0,0.358732,0.0,0.0,0.0,0.0,0.217256,0.0,0.0,0.074224,0.0,0.113689 };
*/
int tree[1000][2][1000];


//int Bmatrix[Def_Bk][Def_Bn] = { 3,3,3,3,3,3,3,3,3 };       //protograph matrix

int LookUpOnes[GF];

int VNsNeighbors[Def_n][Def_dv];
int CNsNeighbors[Def_nk][Def_dc];

int VN_degree[Def_n] = { 0 };
int CN_degree[Def_nk] = { 0 };

int VNtoCNconnection[Def_n][Def_dv];
int CNtoVNconnection[Def_nk][Def_dc];

int codeword_ref[Def_n] = { 0 };

double mVNtoCN[Def_n][Def_dv][GF];
double mCNtoVN[Def_nk][Def_dc][GF];

int perm_code[Def_n];
int permutationH[Def_n];
int permutationH2[3 * Def_n];
int permutationH3[Def_Bn][Def_n];

int perm_sparse[1024];

int offset_LDPC[Def_n];

double jaclog_lookup[MAXS_H_NUMSAMPLES];


int mult_table[GF][GF] = { {0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0},
						   {0,	1,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,	14,	15},
						   {0,	2,	4,	6,	8,	10,	12,	14,	3,	1,	7,	5,	11,	9,	15,	13},
						   {0,	3,	6,	5,	12,	15,	10,	9,	11,	8,	13,	14,	7,	4,	1,	2},
						   {0,	4,	8,	12,	3,	7,	11,	15,	6,	2,	14,	10,	5,	1,	13,	9},
						   {0,	5,	10,	15,	7,	2,	13,	8, 14,	11,	4,	1,	9,	12,	3,	6},
						   {0,	6,	12,	10,	11,	13,	7,	1,	5,	3,	9,	15,	14,	8,	2,	4},
						   {0,	7,	14,	9,	15,	8,	1,	6,	13,	10,	3,	4,	2,	5,	12,	11},
						   {0,	8,	3,	11,	6,	14,	5,	13,	12,	4,	15,	7,	10,	2,	9,	1},
						   {0,	9,	1,	8,	2,	11,	3,	10,	4,	13,	5,	12,	6,	15,	7,	14},
						   {0,	10,	7,	13,	14,	4,	9,	3,	15,	5,	8,	2,	1,	11,	6,	12},
						   {0,	11,	5,	14,	10,	1,	15,	4,	7,	12,	2,	9,	13,	6,	8,	3},
						   {0,	12,	11,	7,	5,	9,	14,	2,	10,	6,	1,	13,	15,	3,	4,	8},
						   {0,	13,	9,	4,	1,	12,	8,	5,	2,	15,	11,	6,	3,	14,	10,	7},
						   {0,	14,	15,	1,	13,	3,	2,	12,	9,	7,	6,	8,	4,	10,	11,	5},
						   {0,	15,	13,	2,	9,	6,	4,	11,	1,	14,	12,	3,	8,	7,	5,	10} };

/*
int mult_table[GF][GF] = { {0,	0,	0,	0,	0,	0,	0,	0},
						   {0,	1,	2,	3,	4,	5,	6,	7},
						   {0,	2,	4,	6,	3,	1,	7,	5},
						   {0,	3,	6,	5,	7,	4,	1,	2},
						   {0,	4,	3,	7,	6,	2,	5,	1},
						   {0,	5,	1,	4,	2,	7,	3,	6},
						   {0,	6,	7,	1,	5,	3,	2,	4},
						   {0,	7,	5,	2,	1,	6,	4,	3} };
*/
/*
int mult_table[GF][GF] = { {0, 0, 0, 0 },
						   {0, 1, 2, 3 },
						   {0, 2, 3, 1 },
		  				   {0, 3, 1, 2 } };
*/

int add_table[GF][GF] = { {0,	1,	2,	3,	4,	5,	6,  7,	8,	9,	10, 11,	12,	13,	14,	15},
						   {1,	0,	3,	2,	5,	4,	7,	6,	9,	8,	11,	10,	13,	12,	15,	14},
						   {2,	3,	0,	1,	6,	7,	4,	5,	10,	11,	8,	9,	14,	15,	12,	13},
						   {3,	2,	1,	0,	7,	6,	5,	4,	11,	10,	9,	8,	15,	14,	13,	12},
						   {4,	5,	6,  7,	0,	1,	2,	3,	12,	13,	14,	15,	8,	9,	10, 11},
						   {5,	4,	7,	6,	1,	0,	3,	2,  13,	12,	15,	14,	9,	8,	11,	10},
						   {6,	7,	4,	5,	2,	3,	0,	1,	14,	15,	12,	13,	10,	11,	8,	9},
						   {7,	6,	5,	4,	3,	2,	1,	0,	15,	14,	13,	12,	11,	10,	9,	8},
						   {8,	9,	10,	11,	12,	13,	14,	15,	0,	1,	2,	3,	4,	5,	6,	7},
						   {9,	8,	11,	10,	13,	12, 15,	14,	1,	0,	3,	2,	5,	4,	7,	6},
						   {10,	11,	8,	9,	14,	15,	12,	13,	2,	3,	0,	1,	6,	7,	4,	5},
						   {11,	10,	9,	8,	15,	14,	13,	12,	3,	2,	1,	0,	7,	6,	5,	4},
						   {12,	13,	14,	15,	8,	9,	10,	11,	4,	5,	6,	7,	0,	1,	2,	3},
						   {13,	12,	15,	14,	9,	8,	11,	10,	5,	4,	7,	6,	1,	0,	3,	2},
						   {14,	15,	12,	13,	10,	11,	8,	9,	6,	7,	4,	5,	2,	3,	0,	1},
						   {15,	14,	13,	12,	11,	10,	9,	8,	7,	6,	5,	4,	3,	2,	1,	0} };

/*
int add_table[GF][GF] = { {0,	1,	2,	3,	4,	5,	6,  7},
						   {1,	0,	3,	2,	5,	4,	7,	6},
						   {2,	3,	0,	1,	6,	7,	4,	5},
						   {3,	2,	1,	0,	7,	6,	5,	4},
						   {4,	5,	6,  7,	0,	1,	2,	3},
						   {5,	4,	7,	6,	1,	0,	3,	2},
						   {6,	7,	4,	5,	2,	3,	0,	1},
						   {7,	6,	5,	4,	3,	2,	1,	0} };
*/

/*
int add_table[GF][GF] = { {0, 1, 2, 3 },
						   {1, 0, 3, 2 },
						   {2, 3, 0, 1 },
						   {3, 2, 1, 0 } };
*/
/*
int div_table[GF][GF] = { {1,    9,   14,   13,   11,    7,    6,   15 ,   2,   12,    5,   10,    4,    3,    8},
							{2,    1,   15,    9,    5,  14,   12,   13,    4,  11,   10,    7,    8,    6,    3},
							{3,   8,   1,    4,   14,    9,   10,    2,    6,    7,   15,   13,   12,    5,   11},
							{4,    2,   13,    1,   10,   15,   11,    9,    8,  5,    7,   14,    3,   12,    6},
							{5,   11,    3,   12,    1,    8,   13,    6,   10,  9,    2,    4,    7,   15,   14},
							{6,    3,    2,    8,   15,    1,    7,    4,   12, 14,   13,    9,   11,   10,    5},
							{7,   10,   12,    5,    4,    6,    1,   11,   14,  2,    8,    3,   15,    9,   13},
							{8,    4,    9,    2,    7,   13,    5,    1,    3, 10,   14,   15,    6,   11,   12},
							{9,   13,    7,   15,   12,   10,    3,   14,    1,  6,   11,    5,    2,    8,    4},
							{10,    5,    6,   11,    2,    3,    9,   12,   7,  1,    4,    8,   14,   13,   15},
							{11,   12,    8,    6,   9,    4,    15,    3,   5, 13,    1,    2,   10,   14,    7},
							{12,    6,    4,    3,   13,    2,   14,    8,  11, 15,    9,    1,    5,    7,   10},
							{13,   15,   10,   14,    6,    5,    8,    7,   9,  3,   12,   11,    1,    4,    2},
							{14,    7,   11,   10,    8,   12,    2,    5,   15, 4,    3,   6,   13,    1,    9},
							{15,   14,    5,    7,    3,   11,    4,   10,   13,  8,    6,   12,    9 ,   2,    1},
};
*/
/*
int mult_table[GF][GF] = { {0,0,0,0,0,0,0,0},
						   {0,1,2,3,4,5,6,7},
						   {0,2,4,6,3,1,7,5},
						   {0,3,6,5,7,4,1,2},
						   {0,4,3,7,6,2,5,1},
						   {0,5,1,4,2,7,3,6},
						   {0,6,7,1,5,3,2,4},
						   {0,7,5,2,1,6,4,3} };
*/

int add_tableDNA[4][4] = { {0, 1, 2, 3 },
						   {1, 0, 3, 2 },
						   {2, 3, 0, 1 },
						   {3, 2, 1, 0 } };

typedef struct
{
	int nedge;
	int trellis[20000000][2 * Rep_Factor + 2];
} TrellisStruct;

TrellisStruct Codetrellis;

int LookUpOnesSparse[256];
int LookUpSparse[inner_codebook_numb][GF] = { 0 };
int watermark_strings[added_sequence_numb][Def_nw] = { 0 };

int state_mapper[2000][Rep_Factor] = { 0 };

double Forward[2000][Def_n * Def_nw + 1] = { 0 };
double Backward[2000][2] = { 0 };

//vector<vector<vector<double>>> Middle(pow(2, Def_nw), vector<vector<double>>(Def_nw / 2 * (Ins_max + 1) + 1, vector<double>(pow(4, Def_nw / 2 * (Ins_max + 1)), -1)));
//vector<vector<vector<double>>> Middle((Def_n* Def_nw / 2) * 2 + 1, vector<vector<double>>(pow(2, Def_nw), vector<double>(Def_nw / 2 * (Ins_max + 1) + 1)));

//vector<vector<double>>Forward(200000, vector<double>(Def_n * Def_nw + 1,0));

vector<vector<vector<vector<double>>>> Middle(Rep_Factor, vector<vector<vector<double>>>((Def_n* Def_nw / 2) * 2 + 1, vector<vector<double>>(pow(2, Def_nw), vector<double>(Def_nw / 2 * (Ins_max + 1) + 1))));

double Liklehoods[Rep_Factor][GF][Def_n];

double Liklehoods_test[Rep_Factor][4][Def_n * Def_nw / 2];

/*************************************************************************/
/**************************   GENREAL VARIABLES   ************************/
/*************************************************************************/

unsigned long int l;                  /* 32 celle LFSR */
unsigned long int l_start;            /* valore iniziale LFSR */

unsigned long int l_orig;                  /* 32 celle LFSR */
unsigned long int l_start_orig;            /* valore iniziale LFSR */

long int vc_un;                       /* variabile intera 31 bit per    */
long int vc_unInterH;                       /* variabile intera 31 bit per    */
long int vc_unInter;                       /* variabile intera 31 bit per    */
									  /* la variabile uniforme */
long int vc_un_start;                 /* valore iniziale */
long int vc_unch;

long int vc_un_orig;                       /* variabile intera 31 bit per    */
long int vc_unInterH_orig;                       /* variabile intera 31 bit per    */
long int vc_unInter_orig;                       /* variabile intera 31 bit per    */
									  /* la variabile uniforme */
long int vc_un_start_orig;                 /* valore iniziale */
long int vc_unch_orig;

int frame;
int bit_err;
int symb_err;
int frame_err, numb_frame_err;

double P_ins;                         //Probability of insertion
double P_del;                         //Probability of deletion
double P_subs;                        //Probability of substitution
double P_trans;                       //Probability of transmission

int drift_max;
int states_max;

int ncheck;                           /* periodo risultati intermedi */

int MaxNumIt;
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
	s = vc_unInter / q; vc_unInter;
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

int willIstop()
{
	if (frame_err < numb_frame_err) return 0;

	return 1;
}

/*************************************************************************/
/***************   Initialize variables and simulation   *****************/
/*************************************************************************/

// Utility function to find minimum of three numbers 
int minimum(int x, int y, int z)
{
	return min(min(x, y), z);
}

int editDist(int* str1, int* str2, int m, int n)
{
	int i, j, t, track;
	int dist[Def_nw + 1][Def_nw + 1];

	// If first string is empty, the only option is to 
	// insert all characters of second string into first 
	if (m == 0)
		return n;

	// If second string is empty, the only option is to 
	// remove all characters of first string 
	if (n == 0)
		return m;


	for (i = 0; i <= m; i++)
	{
		dist[0][i] = i;
	}
	for (j = 0; j <= n; j++)
	{
		dist[j][0] = j;
	}

	for (j = 1; j <= m; j++) {
		for (i = 1; i <= n; i++) {
			if (str1[i - 1] == str2[j - 1]) {
				track = 0;
			}
			else {
				track = 1;
			}
			t = min((dist[i - 1][j] + 1), (dist[i][j - 1] + 1));
			dist[i][j] = min(t, (dist[i - 1][j - 1] + track));
		}
	}

	return(dist[m][n]);
}


/////////////////////////////
/// Calculate receiver metric
/////////////////////////////
double Receiver_metric(int* temp_recieved, int* temp_trans, int length1, int length2)
{
	int i, j;
	double F_pq[100][100];

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

void initialize_variables(int n)
{
	int i, j, ipick;
	int GF_sparse = pow(2, Def_nw);
	int temp = 0;

	/***************************   NUMERO DI FRAME   *************************/

	l_start = Def_lstart;
	l = Def_lstart;
	vc_un = vc_un_orig;
	vc_unInter = vc_unInter_orig;
	vc_unInterH = vc_unInterH_orig;
	vc_unch = vc_unch_orig;

	MaxNumIt = Def_MaxNumIt;

	ncheck = Def_CheckFrames;

	for (i = 0; i < n; i++)
		permutationH[i] = i;

	for (i = 0; i < Def_nk; i++)
		for (j = 0; j < Def_Bk; j++)
			permutationH3[j][i] = i;

	for (i = 0; i < 3 * n; i++)
		permutationH2[i] = i;

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

	printf("Number of frames in error to validate a simulation point: %d \n", numb_frame_err);

}

/////////////////////////////////
/// Calculate change probability
/////////////////////////////////
void Change_Probability(int nw, double& P_C, int codebook[GF])
{
	int i, j, ii, jj;

	int str1[Def_nw] = { 0 };
	int str2[Def_nw] = { 0 };

	for (i = 0; i < GF; i++)
	{
		for (ii = 0; ii < Def_nw; ii++)
			str1[ii] = (codebook[i] >> ii) & 1LL;

		for (j = 0; j < GF; j++)
		{
			if (j != i)
			{
				for (jj = 0; jj < Def_nw; jj++)
					str2[jj] = (codebook[j] >> jj) & 1LL;

				P_C += Receiver_metric(str1, str2, Def_nw, Def_nw);
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
/// Build inner code trellis
/////////////////////////////
void Build_inner_codetrellis(int nw, int states_max)
{
	int count = 0;
	int cond = 0;
	int rep = Rep_Factor;

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
				Codetrellis.trellis[count][0] = x_1;
				for (int i = 0; i < rep; i++)
					Codetrellis.trellis[count][i + 1] = state_mapper[x_1][i];

				Codetrellis.trellis[count][rep + 1] = x_2;
				for (int i = 0; i < rep; i++)
					Codetrellis.trellis[count][rep + 1 + i + 1] = state_mapper[x_2][i];

				count++;
			}
		}
	}

	Codetrellis.nedge = count;

}
/////////////////////////////
/// Build inner code trellis
/////////////////////////////


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

/////////////////////////
/// Sorting Algorithm
/////////////////////////
// Function to swap two pointers
void swap(double* a, double* b)
{
	double temp = *a;
	*a = *b;
	*b = temp;
}

void swap_int(int* a, int* b)
{
	int temp = *a;
	*a = *b;
	*b = temp;
}

void quicksort(double arr[], int l, int r, int sorted[])
{
	// Base case: No need to sort arrays of length <= 1
	if (l >= r)
	{
		return;
	}

	// Choose pivot to be the last element in the subarray
	double pivot = arr[r];

	// Index indicating the "split" between elements smaller than pivot and 
	// elements greater than pivot
	int cnt = l;

	// Traverse through array from l to r
	for (int i = l; i <= r; i++)
	{
		// If an element less than or equal to the pivot is found...
		if (arr[i] >= pivot)
		{
			// Then swap arr[cnt] and arr[i] so that the smaller element arr[i] 
			// is to the left of all elements greater than pivot
			swap(&arr[i], &arr[cnt]);
			swap_int(&sorted[i], &sorted[cnt]);
			// Make sure to increment cnt so we can keep track of what to swap
			// arr[i] with 
			cnt++;
		}
	}

	quicksort(arr, l, cnt - 2, sorted); // Recursively sort the left side of pivot
	quicksort(arr, cnt, r, sorted);   // Recursively sort the right side of pivot
}
/////////////////////////
/// Sorting Algorithm
/////////////////////////


/////////////////////////////
/// Choose n best substrings
/////////////////////////////
void best_added_substrings(int nw, int numb)
{
	int i, j, ii;

	double P_ref = 0;
	double P_C = 0;
	double dev[1024];
	int LookUpSparse_ref[GF] = { 0 };
	int GF_sparse = pow(2, Def_nw);

	Change_Probability(nw, P_ref, LookUpSparse[0]);

	int str1[Def_nw] = { 0 };
	int str2[Def_nw] = { 0 };

	for (i = 0; i < GF_sparse; i++)
	{
		P_C = 0;

		for (ii = 0; ii < nw; ii++)
			str2[ii] = (i >> ii) & 1LL;

		for (j = 0; j < GF; j++)
		{
			LookUpSparse_ref[j] = 0;

			for (ii = 0; ii < nw; ii++)
				str1[ii] = (LookUpSparse[0][j] >> ii) & 1LL;

			for (ii = 0; ii < nw; ii++)
				LookUpSparse_ref[j] += (str1[ii] ^ str2[ii]) << ii;
		}

		Change_Probability(nw, P_C, LookUpSparse_ref);

		dev[i] = (P_ref - P_C) / P_ref;
	}

	int sorted[1024] = { 0 };

	for (i = 0; i < GF_sparse; i++)
		sorted[i] = i;

	quicksort(dev, 0, GF_sparse - 1, sorted);

	for (i = 0; i < numb; i++)
	{
		for (j = 0; j < nw; j++)
			watermark_strings[i][j] = (sorted[i] >> j) & 1LL;
	}
}
/////////////////////////////
/// Choose n best substrings
/////////////////////////////

/*
////////////////////////////////////////
/// Pre-compute branch/receiver metrics
////////////////////////////////////////
void compute_BranchMetrics(int nw, int q)
{
	int i, j, ii;
	int max_lengthY = nw * (Ins_max + 1);

	int* x = new int[nw]; //transmitted symbol
	int* y = new int[max_lengthY]; //Received sequence

	int y_int; //decimal value of received sequence

	for (i = 0; i < pow(q, nw); i++)
	{//loop over all possible transmitted symbols X

		int2vec_q(x, i, nw, q);

		for (j = 0; j <= max_lengthY; j++)
		{//loop over all possible lengths of received sequence Y
			for (ii = 0; ii < pow(4, j); ii++)
			{
				int2vec_q(y, ii, max_lengthY, q); //convert int reprsentation of Y to a vect representation

				if (inner_decoder_domain == 0)
					Middle[i][j][ii] = Receiver_metric(y, x, j, nw);
				else
					Middle[i][j][ii] = log(Receiver_metric(y, x, j, nw));
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

void initialize_sim(int n, int nw, int sim)
{
	int i, j;
	int rep = Rep_Factor;

	int nw_base = 2;

	l_start = l_start_orig;
	l = l_orig;
	vc_un = vc_un_orig;
	vc_unInter = vc_unInter_orig;
	vc_unInterH = vc_unInterH_orig;
	vc_unch = vc_unch_orig;

	bit_err = 0;
	symb_err = 0;
	frame_err = 0;

	P_ins = Def_InsDelProbIni - sim * Def_InsDelProbDelta;
	P_del = Def_InsDelProbIni - sim * Def_InsDelProbDelta;
	P_trans = 1 - P_ins - P_del;
	P_subs = Def_SubsProb;

	drift_max = 10 * sqrt(n * nw_base / 2 * P_del / (1 - P_del));
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

	Build_inner_codetrellis(nw_base / 2, states_max);

	//Build_inner_codetrellis(nWM, drift_max);

	if (added_sequence_type == 2)
		best_added_substrings(nw / 2, added_sequence_numb);
}


/////////////////////////////
/// Build inner codebook
/////////////////////////////
void build_inner_codebook(int nw, int design)
{
	int i, j, ii, ipick;
	int weight = 1;
	int temp;
	int GF_sparse = pow(2, Def_nw);
	int check_count[Def_nw + 1] = { 0 };

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
		int codeword[Def_nw] = { 0 };
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
		Change_Probability(nw, P_C, LookUpSparse[0]);

		P_C_init = P_C;

		while (E_unchanged < 10)
		{
			while (iter < iter_max && E_drops < E_drops_max)
			{
				numb = 0;

				pertub_c = unif_int() % GF;
				pertub_c_temp = LookUpSparse[0][pertub_c];
				pertub_b = unif_int() % nw;

				for (i = 0; i < nw; i++)
					codeword[i] = (LookUpSparse[0][pertub_c] >> i) & 1LL;

				codeword[pertub_b] = codeword[pertub_b] ^ 1;

				for (i = 0; i < nw; i++)
					numb += codeword[i] << i;

				//Change from C to C'
				LookUpSparse[0][pertub_c] = numb;

				//Calculate the energy of the chosen code C'
				Change_Probability(nw, P_C_prime, LookUpSparse[0]);

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
		Change_Probability(nw, P_C, LookUpSparse[0]);
	}
	else if (design == 2)
	{
		//int manual_codebook[GF] = { 0,60,26,38,41,21,51,15 }; //HD 1
		//int manual_codebook[GF] = { 32,24,42,6,9,57,27,39 }; //LD 1
		//int manual_codebook[GF] = { 52,12,2,58,33,25,23,47 }; //HD 2
		//int manual_codebook[GF] = { 0,56,12,42,33,19,15,63 }; //LD 2

		int manual_codebook[GF] = { 0,1,2,3 };

		for (i = 0; i < GF; i++)
			LookUpSparse[0][i] = manual_codebook[i];
	}
	else if (design == 3)
	{
		/*
			int manual_n_codebook[inner_codebook_numb][GF] = { {0,44,21,57,18,42,7,63 },
				{0,28,21,33,54,42,11,63} };
			*/
			/*
			int manual_n_codebook[inner_codebook_numb][GF] = { {0,160,236,17,85,245,185,98,38,170,250,115,55,11,175,255},
	{64,240,152,33,165,117,205,130,202,90,222,19,171,59,95,191} };
			*/
			/*
		int manual_n_codebook[inner_codebook_numb][GF] = { {0,160,236,17,85,245,185,98,38,170,250,115,55,11,175,255},
{64,240,152,33,165,117,205,130,202,90,222,19,171,59,95,191},
{48,84,184,220,65,137,237,10,218,170,206,131,155,95,47,255},
{192,160,188,81,5,245,185,102,10,234,206,67,55,139,123,255} };
			*/

		int manual_n_codebook[inner_codebook_numb][GF] = {0};

		for (i = 0; i < inner_codebook_numb; i++)
			for (j = 0; j < GF; j++)
				LookUpSparse[i][j] = manual_n_codebook[i][j];
	}
	else if (design == 4)
	{
		int codeword[Def_nw / 2] = { 0 };
		int sum1 = 0, sum2 = 0, cnt = 0;
		int VT_codebook[200] = { 0 };
		//int As[Def_nWM / 2 - 1] = { 0 };
		int As[Def_nw / 2] = { 0 };

		for (i = 0; i < pow(4, nw / 2); i++)
		{
			int2vec_q(codeword, i, nw / 2, 4);

			for (j = 0; j < nw / 2 - 1; j++)
				As[j] = 0;
			for (j = 0; j < nw / 2 - 1; j++)
				if (codeword[j + 1] >= codeword[j])
					As[j] = 1;

			sum1 = 0;
			sum2 = 0;

			for (j = 0; j < nw / 2; j++)
				sum1 += codeword[j];

			for (j = 0; j < nw / 2 - 1; j++)
				sum2 += (j + 1) * As[j];

			sum1 %= 4;
			sum2 %= (nw / 2 + 1);

			if (sum1 == 0 && sum2 == a_VT)
				VT_codebook[cnt++] = i;
		}

		for (j = 0; j < GF; j++)
			for (i = 0; i < inner_codebook_numb; i++)
				LookUpSparse[i][j] = VT_codebook[i * GF + j];
		//LookUpSparse[i ^ j%2][j] = VT_codebook[i * GF + j];
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


void print_intermediate_results(int n, double P_ins, double P_del, double P_subs, int sim)
{
	char ligne[200], chaine[500];

	printf("\nIntermediate Results (frame no. %d), Sim = %d, frames err = %d, frames = %d\n", (frame + 1), sim, frame_err, frame + 1);
	//printf("load = %d\n",ErrorVectorWeight);
	printf("%e %e %e %e %e\n", P_ins, P_subs, (double)bit_err / ((frame + 1) * n * log2(GF)), (double)symb_err / ((frame + 1) * n), (double)frame_err / (frame + 1));
}


void printresults(int sim, int n, int nk, int nw, int nkw, int dv, int dc, double P_ins, double P_del, double P_subs, int thread)
{
	FILE* fout;

	char stringa[500];

	frame++;

	if (output_type == 0)
	{
		printf("%e %e %e %e %e %e\n", P_ins, P_del, P_subs, (double)bit_err / (frame * n * log2(GF)), (double)symb_err / (frame * n), (double)frame_err / frame);

		sprintf(stringa, "Davey_MacKay_Construction_NonBinary_Wimax_LDPC_FFTSPA_DNA_alphabet_BERFER_n%d_k%d_dv%d_dc%d_GF%d_WaterMark_nw%d_kw%d.txt", n, n - nk, dv, dc, GF, nw, nkw);
		if (sim == 0)
		{
			fout = fopen(stringa, "w");
			fprintf(fout, "pins pdel psubs BER SER FER\n");
		}
		else
			fout = fopen(stringa, "a");
		fprintf(fout, "%e %e %e %e %e %e\n", P_ins, P_del, P_subs, (double)bit_err / (frame * n * log2(GF)), (double)symb_err / (frame * n), (double)frame_err / frame);
		fclose(fout);
	}
	else
	{
		sprintf(stringa, "Output parallel/Davey_MacKay_Construction_NonBinary_(3,6)_LDPC_FFTSPA_DNA_alphabet_BERFER_n%d_k%d_dv%d_dc%d_GF%d_Watermark_nw%d_kw%d_TVC_it/thread_%d.txt", n, n - nk, dv, dc, GF, nw, nkw, thread);
		//sprintf(stringa, "Output parallel/Davey_MacKay_Construction_NonBinary_(3,6)_LDPC_FFTSPA_DNA_alphabet_BERFER_n%d_k%d_dv%d_dc%d_GF%d_Watermark_nw%d_kw%d_TVC_rep%d/thread_%d.txt", n, n - nk, dv, dc, GF, nw, nkw, Rep_Factor, thread);
		//sprintf(stringa, "Output parallel/test/thread_%d.txt", thread);
		if (sim == 0)
		{
			fout = fopen(stringa, "w");
			//fprintf(fout, "pins pdel psubs BER SER FER\n");
		}
		else
			fout = fopen(stringa, "a");
		fprintf(fout, "%e %e %e %d %d %d\n", P_ins, P_del, P_subs, bit_err, symb_err, frame);
		fclose(fout);
	}
}


///////////////////////////////////////////
///Progressive-edge growtj (PEG) algorithm
///////////////////////////////////////////
void PEG_algorithm(int n, int nk, int dv, int dc)
{
	int i, j, ii, jj, k, VN, CN;
	int VN_temp, CN_temp;
	int CN_deg[Def_nk] = { 0 };
	int VN_neighborhood[Def_nk] = { 0 };
	int available_CN[Def_nk] = { 0 };
	int same_min_deg_CN[Def_nk] = { 0 };
	int same_deg_cnt = 0;
	int VN_deg[Def_n] = { 0 };
	int depth = 0;
	int sum;
	int cnt_CN[1000] = { 0 };
	int cnt_VN[1000] = { 0 };
	int CN_degree_min, min_degree;

	for (i = 0; i < nk; i++)
		CN_deg[i] = 0;

	for (i = 0; i < n; i++)
	{//loop over all VNs
		VN_deg[i] = 0;

		for (j = 0; j < VN_degree[i]; j++)
		{//loop over the dregree of VN i
			if (j == 0)
			{//if this is the first edge of VN i

				CN_degree_min = 0;
				min_degree = 100;

				same_deg_cnt = 0;

				for (ii = 0; ii < nk; ii++)
				{
					if (CN_deg[ii] < min_degree && available_CN[ii] == 0)
					{
						min_degree = CN_deg[ii];
						CN_degree_min = ii;
						same_deg_cnt = 0;
					}

					if (CN_deg[ii] == min_degree && available_CN[ii] == 0)
					{
						same_min_deg_CN[same_deg_cnt] = ii;
						same_deg_cnt++;
					}
				}

				if (same_deg_cnt > 0)
					CN_degree_min = same_min_deg_CN[unif_int() % same_deg_cnt];

				VNsNeighbors[i][j] = CN_degree_min;
				VN_deg[i]++;
				CNsNeighbors[CN_degree_min][CN_deg[CN_degree_min]++] = i;

				if (CN_deg[CN_degree_min] >= CN_degree[CN_degree_min])
					available_CN[CN_degree_min] = 1;
			}
			else
			{//if this is not the first edge of VN i

				for (ii = 0; ii < 1000; ii++)
				{
					cnt_VN[ii] = 0;
					cnt_CN[ii] = 0;
				}

				for (ii = 0; ii < 1000; ii++)
					for (jj = 0; jj < 2; jj++)
						for (k = 0; k < 1000; k++)
							tree[ii][jj][k] = -1;

				depth = 0;

				tree[depth][0][cnt_CN[depth]] = i;
				VN_temp = i;

				for (ii = 0; ii < nk; ii++)
					VN_neighborhood[ii] = 0;

				cnt_VN[depth]++;

				//Expan tree from root node
				for (ii = 0; ii < VN_deg[VN_temp]; ii++)
				{
					tree[depth][1][cnt_CN[depth]++] = VNsNeighbors[VN_temp][ii];
					VN_neighborhood[VNsNeighbors[VN_temp][ii]]++;
				}

				while (true)
				{
					//Expan tree from CNs

					for (ii = 0; ii < cnt_CN[depth]; ii++)
					{
						CN_temp = tree[depth][1][ii];

						for (jj = 0; jj < CN_deg[CN_temp]; jj++)
						{
							int cnt = 0;

							for (k = 0; k < cnt_VN[depth]; k++)
								if (tree[depth][0][k] == CNsNeighbors[CN_temp][jj])
									cnt++;

							if (cnt > 0 || CNsNeighbors[CN_temp][jj] == i)
								continue;

							cnt = 0;

							for (k = 0; tree[depth + 1][0][k] != -1; k++)
								if (CNsNeighbors[CN_temp][jj] == tree[depth + 1][0][k])
									cnt++;

							if (cnt > 0)
								continue;

							tree[depth + 1][0][cnt_VN[depth + 1]++] = CNsNeighbors[CN_temp][jj];
						}
					}

					//Expan tree from VNs

					depth++;

					for (ii = 0; ii < cnt_VN[depth]; ii++)
					{
						VN_temp = tree[depth][0][ii];

						for (jj = 0; jj < VN_deg[VN_temp]; jj++)
						{

							int cnt = 0;

							for (k = 0; k < cnt_CN[depth - 1]; k++)
								if (tree[depth - 1][1][k] == VNsNeighbors[VN_temp][jj] || VN_neighborhood[VNsNeighbors[VN_temp][jj]] == 1)
									cnt++;

							if (cnt > 0)
								continue;

							cnt = 0;

							for (k = 0; tree[depth][1][k] != -1; k++)
								if (VNsNeighbors[VN_temp][jj] == tree[depth][1][k])
									cnt++;

							if (cnt > 0)
								continue;

							tree[depth][1][cnt_CN[depth]++] = VNsNeighbors[VN_temp][jj];
							VN_neighborhood[VNsNeighbors[VN_temp][jj]]++;
						}
					}

					sum = 0;

					for (ii = 0; ii < nk; ii++)
					{
						if (available_CN[ii] == 0)
							sum += VN_neighborhood[ii];
						else
							sum += 1;
					}

					if ((cnt_CN[depth] == 0 || cnt_VN[depth] == 0) && sum >= nk)
					{
						depth -= 1;

						CN_degree_min = 10000;
						min_degree = 100;

						same_deg_cnt = 0;

						for (ii = 0; ii < cnt_CN[depth]; ii++)
						{
							CN_temp = tree[depth][1][ii];

							if (CN_deg[CN_temp] < min_degree && available_CN[CN_temp] == 0)
							{
								min_degree = CN_deg[CN_temp];
								CN_degree_min = CN_temp;
								same_deg_cnt = 0;
							}

							if (CN_deg[CN_temp] == min_degree && available_CN[CN_temp] == 0)
							{
								same_min_deg_CN[same_deg_cnt] = CN_temp;
								same_deg_cnt++;
							}
						}

						if (same_deg_cnt > 0)
						{
							CN_degree_min = same_min_deg_CN[unif_int() % same_deg_cnt];

							VNsNeighbors[i][j] = CN_degree_min;
							VN_deg[i]++;
							CNsNeighbors[CN_degree_min][CN_deg[CN_degree_min]++] = i;

							if (CN_deg[CN_degree_min] >= CN_degree[CN_degree_min])
								available_CN[CN_degree_min] = 1;
						}
						else
						{
							CN_degree_min = 10000;
							min_degree = 100;

							same_deg_cnt = 0;

							for (ii = 0; ii < nk; ii++)
							{
								int cnt = 0;

								for (k = 0; k < cnt_CN[0]; k++)
									if (ii == tree[0][1][k])
										cnt++;

								if (CN_deg[ii] < min_degree && available_CN[ii] == 0 && cnt < 1)
								{
									min_degree = CN_deg[ii];
									CN_degree_min = ii;
								}

								if (CN_deg[ii] == min_degree && available_CN[ii] == 0 && cnt < 1)
								{
									same_min_deg_CN[same_deg_cnt] = ii;
									same_deg_cnt++;
								}
							}

							if (same_deg_cnt > 0)
								CN_degree_min = same_min_deg_CN[unif_int() % same_deg_cnt];

							if (CN_degree_min == 10000)
								CN_degree_min = unif_int() % (nk);

							VNsNeighbors[i][j] = CN_degree_min;
							VN_deg[i]++;
							CNsNeighbors[CN_degree_min][CN_deg[CN_degree_min]++] = i;

							if (CN_deg[CN_degree_min] >= CN_degree[CN_degree_min])
								available_CN[CN_degree_min] = 1;
						}
						break;
					}
					else if ((cnt_CN[depth] == 0 || cnt_VN[depth] == 0) && sum < nk)
					{
						CN_degree_min = 0;
						min_degree = 100;

						same_deg_cnt = 0;

						for (ii = 0; ii < nk; ii++)
						{
							if (CN_deg[ii] < min_degree && available_CN[ii] == 0)
							{
								min_degree = CN_deg[ii];
								CN_degree_min = ii;
								same_deg_cnt = 0;
							}

							if (CN_deg[ii] == min_degree && available_CN[ii] == 0)
							{
								same_min_deg_CN[same_deg_cnt] = ii;
								same_deg_cnt++;
							}
						}

						if (same_deg_cnt > 0)
							CN_degree_min = same_min_deg_CN[unif_int() % same_deg_cnt];

						VNsNeighbors[i][j] = CN_degree_min;
						VN_deg[i]++;
						CNsNeighbors[CN_degree_min][CN_deg[CN_degree_min]++] = i;

						if (CN_deg[CN_degree_min] >= CN_degree[CN_degree_min])
							available_CN[CN_degree_min] = 1;

						break;
					}
				}
			}
		}
	}

	int compliance = 0;

	for (i = 0; i < n; i++)
		VN_degree[i] = VN_deg[i];


	for (i = 0; i < nk; i++)
		compliance += abs(CN_degree[i] - CN_deg[i]);

	for (i = 0; i < nk; i++)
		CN_degree[i] = CN_deg[i];
}

///////////////////////////////////////////
///Progressive-edge growtj (PEG) algorithm
///////////////////////////////////////////



///////////////////////////////////////////////////////
///Build parity check matrix of NB-LDPC code
///////////////////////////////////////////////////////

void build_paritycheckmatrix_v02(int n, int nk, int Bnk, int Bn, int dv, int dc, int type, int sim)
{
	ifstream inFile;
	FILE* fout;
	int x;

	int i, j, m, k, ii, jj;
	int ipick, temp;
	int dv_temp, dc_temp;

	int nr_submatrix; // number of rows of each cyclic submatrix
	int nc_submatrix; // number of columns of each cyclic submatrix
	int degree;
	int Neighbor;
	int NumCNNeighbor;

	nc_submatrix = n / Bn;
	nr_submatrix = nk / Bnk;

	int* row = new int[nc_submatrix];
	int* firstrow = new int[nc_submatrix];
	int* interleaver = new int[nc_submatrix];

	int NumVNNeighbor[Def_n] = { 0 };
	int Neigh[Def_Bk][Def_Bn][Def_dc];

	for (i = 0; i < n; i++)
		VN_degree[i] = Def_dv;

	for (i = 0; i < nk; i++)
		CN_degree[i] = Def_dc;
	/*
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n - nk; j++)
		{
			H[j][i] = 0;
		}
	}
	*/
	if (type == 0)
	{
		for (i = 0; i < Bnk; i++)
		{
			for (j = 0; j < Bn; j++)
			{
				NumCNNeighbor = 0;

				degree = Bmatrix[i][j];
				for (m = 0; m < degree; m++)
					row[m] = 1;
				for (; m < nc_submatrix; m++)
					row[m] = 0;

				//generate the first row of the submatrix, with Bmatrix[i][j] ones randomly placed
				//We start with the vector with all ones at the beginning, i.e., row[], and we interleave it with a random interleaver

				//generate a random interleaver
				for (ii = 0; ii < nc_submatrix; ii++)
				{
					ipick = ii + unif_intH() % (nc_submatrix - ii);
					temp = permutationH[ii];
					permutationH[ii] = permutationH[ipick];
					permutationH[ipick] = temp;
				}

				for (ii = 0; ii < nc_submatrix; ii++)
				{
					interleaver[ii] = permutationH[ii];
					//printf("%d ",interleaver[ii]);
				}

				//create the first row of the submatrix
				for (ii = 0; ii < nc_submatrix; ii++)
					firstrow[ii] = row[interleaver[ii]];

				for (ii = 0; ii < nc_submatrix; ii++)
					if (firstrow[ii] == 1)
					{
						Neigh[i][j][NumCNNeighbor++] = ii;
					}
			}
		}

		for (ii = 0; ii < nr_submatrix; ii++)
			for (i = 0; i < Bnk; i++)
				permutationH3[i][ii] = ii;


		for (ii = 0; ii < nr_submatrix; ii++)
		{
			for (i = 0; i < Bnk; i++)
			{
				ipick = ii + unif_intH() % (nr_submatrix - ii);
				temp = permutationH3[i][ii];
				permutationH3[i][ii] = permutationH3[i][ipick];
				permutationH3[i][ipick] = temp;
			}
		}

		for (k = 0; k < Bnk; k++)
		{
			//first part of the matrix
			for (ii = 0; ii < nr_submatrix; ii++)
			{
				for (jj = 0; jj < dc; jj++)
				{
					ipick = unif_intH() % GF;

					if (ipick == 0)
					{
						jj--;
						continue;
					}
					permutationH2[jj] = ipick;
				}

				NumCNNeighbor = 0;
				for (m = 0; m < Bn; m++)
				{
					for (jj = 0; jj < Bmatrix[k][m]; jj++)
					{
						Neighbor = (Neigh[k][m][jj] + (permutationH3[k][ii] + k * nr_submatrix)) % nc_submatrix + m * nc_submatrix;

						CNsNeighbors[ii + k * nr_submatrix][NumCNNeighbor] = Neighbor;
						VNsNeighbors[Neighbor][NumVNNeighbor[Neighbor]] = ii + k * nr_submatrix;

						CNtoVNconnection[ii + k * nr_submatrix][NumCNNeighbor] = permutationH2[NumCNNeighbor];
						VNtoCNconnection[Neighbor][NumVNNeighbor[Neighbor]] = permutationH2[NumCNNeighbor];

						//H[ii][Neighbor] = permutationH2[NumCNNeighbor];

						NumVNNeighbor[Neighbor] = NumVNNeighbor[Neighbor] + 1;

						NumCNNeighbor++;
					}
				}

				CN_degree[ii + k * nr_submatrix] = NumCNNeighbor;

			}
		}

		for (i = 0; i < n; i++)
			VN_degree[i] = NumVNNeighbor[i];

		//	PEG_algorithm(n, nk, dv, dc);

	}
	else if (type == 1)
	{
		//inFile.open("WiMax_like_N240_K120.txt");

		inFile.open("codes_B1112_1210/AListFormat_Protograph_QC_240.txt");

		//inFile.open("AListFormat_Protograph_QC_240_all1s.txt");

			//inFile.open("wimaxlike_N336_K168_P14_set0.txt");

			//inFile.open("PEGirReg167x334_dv=6_dc=7_WiMax.txt");

			//inFile.open("PEGirReg200x400_dv=10_dc=11_Code1.txt");

		for (i = 0; i < 2; i++)
			inFile >> x;

		inFile >> dv_temp;
		inFile >> dc_temp;

		for (i = 0; i < n; i++)
		{
			inFile >> x;
			VN_degree[i] = x;
		}

		for (i = 0; i < nk; i++)
		{
			inFile >> x;
			CN_degree[i] = x;
		}

		for (i = 0; i < n; i++)
		{
			for (j = 0; j < VN_degree[i]; j++)
			{
				ipick = unif_intH() % GF;

				if (ipick == 0)
				{
					j--;
					continue;
				}
				permutationH2[j] = ipick;
			}

			for (j = 0; j < dv_temp; j++)
			{
				inFile >> x;
				VNsNeighbors[i][j] = x - 1;

				if (j < VN_degree[i])
				{
					VNtoCNconnection[i][j] = permutationH2[j];

					//H[x - 1][i] = permutationH2[j];
				}
			}
		}

		for (i = 0; i < nk; i++)
		{
			for (j = 0; j < dc_temp; j++)
			{
				inFile >> x;
				CNsNeighbors[i][j] = x - 1;

				int temp = 0;

				if (x != 0)
				{
					for (jj = 0; jj < VN_degree[x - 1]; jj++)
					{
						if (VNsNeighbors[x - 1][jj] == i)
						{
							temp = jj;
							break;
						}
					}

					if (j < CN_degree[i])
						CNtoVNconnection[i][j] = VNtoCNconnection[x - 1][temp];
				}
			}
		}

		inFile.close();

	}
	else if (type == 2)
	{
		if (sim == 0)
		{
			for (i = 0; i < Bnk; i++)
			{
				for (j = 0; j < Bn; j++)
				{
					NumCNNeighbor = 0;

					degree = Bmatrix[i][j];
					for (m = 0; m < degree; m++)
						row[m] = 1;
					for (; m < nc_submatrix; m++)
						row[m] = 0;

					//generate the first row of the submatrix, with Bmatrix[i][j] ones randomly placed
					//We start with the vector with all ones at the beginning, i.e., row[], and we interleave it with a random interleaver

					//generate a random interleaver
					for (ii = 0; ii < nc_submatrix; ii++)
					{
						ipick = ii + unif_intH() % (nc_submatrix - ii);
						temp = permutationH[ii];
						permutationH[ii] = permutationH[ipick];
						permutationH[ipick] = temp;
					}

					for (ii = 0; ii < nc_submatrix; ii++)
					{
						interleaver[ii] = permutationH[ii];
						//printf("%d ",interleaver[ii]);
					}

					//create the first row of the submatrix
					for (ii = 0; ii < nc_submatrix; ii++)
						firstrow[ii] = row[interleaver[ii]];

					for (ii = 0; ii < nc_submatrix; ii++)
						if (firstrow[ii] == 1)
						{
							Neigh[i][j][NumCNNeighbor++] = ii;
						}
				}
			}

			for (ii = 0; ii < nr_submatrix; ii++)
				for (i = 0; i < Bnk; i++)
					permutationH3[i][ii] = ii;


			for (ii = 0; ii < nr_submatrix; ii++)
			{
				for (i = 0; i < Bnk; i++)
				{
					ipick = ii + unif_intH() % (nr_submatrix - ii);
					temp = permutationH3[i][ii];
					permutationH3[i][ii] = permutationH3[i][ipick];
					permutationH3[i][ipick] = temp;
				}
			}

			for (k = 0; k < Bnk; k++)
			{
				//first part of the matrix
				for (ii = 0; ii < nr_submatrix; ii++)
				{
					for (jj = 0; jj < dc; jj++)
					{
						ipick = unif_intH() % GF;

						if (ipick == 0)
						{
							jj--;
							continue;
						}
						permutationH2[jj] = ipick;
					}

					NumCNNeighbor = 0;
					for (m = 0; m < Bn; m++)
					{
						for (jj = 0; jj < Bmatrix[k][m]; jj++)
						{
							Neighbor = (Neigh[k][m][jj] + (permutationH3[k][ii] + k * nr_submatrix)) % nc_submatrix + m * nc_submatrix;

							CNsNeighbors[ii + k * nr_submatrix][NumCNNeighbor] = Neighbor;
							VNsNeighbors[Neighbor][NumVNNeighbor[Neighbor]] = ii + k * nr_submatrix;

							CNtoVNconnection[ii + k * nr_submatrix][NumCNNeighbor] = permutationH2[NumCNNeighbor];
							VNtoCNconnection[Neighbor][NumVNNeighbor[Neighbor]] = permutationH2[NumCNNeighbor];

							//H[ii][Neighbor] = permutationH2[NumCNNeighbor];

							NumVNNeighbor[Neighbor] = NumVNNeighbor[Neighbor] + 1;

							NumCNNeighbor++;
						}
					}

					CN_degree[ii + k * nr_submatrix] = NumCNNeighbor;

				}
			}

			for (i = 0; i < n; i++)
				VN_degree[i] = NumVNNeighbor[i];

			PEG_algorithm(n, nk, dv, dc);

			FILE* fout;

			char stringa[500];

			sprintf(stringa, "LDPC_PEG_n%d_nk%d_dv%d_dc%d.txt", n, nk, Def_dv, Def_dc);

			fout = fopen(stringa, "w");
			fprintf(fout, "%d %d\n", n, nk);
			fprintf(fout, "%d %d\n", Def_dv, Def_dc);

			for (i = 0; i < n; i++)
				fprintf(fout, "%d ", VN_degree[i]);

			fprintf(fout, "\n");

			for (i = 0; i < nk; i++)
				fprintf(fout, "%d ", CN_degree[i]);

			fprintf(fout, "\n");

			for (i = 0; i < n; i++)
			{
				for (j = 0; j < VN_degree[i]; j++)
					fprintf(fout, "%d ", VNsNeighbors[i][j] + 1);

				for (; j < Def_dv; j++)
					fprintf(fout, "%d ", VNsNeighbors[i][j]);

				fprintf(fout, "\n");
			}

			for (i = 0; i < nk; i++)
			{
				for (j = 0; j < CN_degree[i]; j++)
					fprintf(fout, "%d ", CNsNeighbors[i][j] + 1);

				for (; j < Def_dc; j++)
					fprintf(fout, "%d ", CNsNeighbors[i][j]);

				fprintf(fout, "\n");
			}

			fclose(fout);
		}
		else
		{
			for (i = 0; i < n; i++)
			{
				for (j = 0; j < VN_degree[i]; j++)
				{
					ipick = unif_intH() % GF;

					if (ipick == 0)
					{
						j--;
						continue;
					}
					VNtoCNconnection[i][j] = ipick;
				}
			}

			for (i = 0; i < nk; i++)
			{
				for (j = 0; j < CN_degree[i]; j++)
				{

					int temp = 0;

					for (jj = 0; jj < VN_degree[CNsNeighbors[i][j]]; jj++)
					{
						if (VNsNeighbors[CNsNeighbors[i][j]][jj] == i)
						{
							temp = jj;
							break;
						}
					}

					CNtoVNconnection[i][j] = VNtoCNconnection[CNsNeighbors[i][j]][temp];
				}
			}
		}
	}
	else
	{
		double num_VN_deg = 0;
		double num_CN_deg = 0;

		double integral;

		int CN_degree_temp[Def_n - Def_nk] = { 0 };

		ii = 0;
		integral = 0;

		for (i = 2; i < Def_dv_irr + 1; i++)
			integral += degree_dist_VN[i] / i;

		for (i = 2; i < Def_dv_irr + 1; i++)
		{
			num_VN_deg = degree_dist_VN[i] / (i * integral) * n;

			for (j = 0; j < num_VN_deg; j++)
				VN_degree[j + ii] = i;

			ii += num_VN_deg;

		}

		if (ii < n)
		{
			for (j = 0; j < n - ii; j++)
				VN_degree[j + ii] = i - 1;

			ii += n - ii;
		}

		ii = 0;
		integral = 0;

		for (i = 2; i < Def_dc_irr + 1; i++)
			integral += degree_dist_CN[i] / i;

		for (i = 2; i < Def_dc_irr + 1; i++)
		{
			num_CN_deg = degree_dist_CN[i] / (i * integral) * (n - nk);

			for (j = 0; j < num_CN_deg; j++)
				CN_degree[j + ii] = i;

			ii += num_CN_deg;
		}

		if (ii < n - nk)
		{
			for (j = 0; j < n - nk - ii; j++)
				CN_degree[j + ii] = i - 1;

			ii += n - nk - ii;
		}

		VN_degree[n - 1] = 0;
		/*
		for (j = 0; j < n; j++)
			VN_degree[j] = Def_dv;

		for (j = 0; j < n - nk; j++)
			CN_degree[j] = Def_dc;
		*/
		//PEG_algorithm(n, nk, dv, dc);

		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n - nk; j++)
				permutationH[j] = j;

			for (j = 0; j < VN_degree[i]; j++)
			{
				int count = 0;

				ipick = j + unif_intH() % ((n - nk) - j);

				for (ii = 0; ii < j; ii++)
					if (ipick == permutationH[ii])
						count++;

				if (CN_degree_temp[ipick] == CN_degree[ipick] || count > 0)
				{
					j--;
					continue;
				}

				temp = permutationH[j];
				permutationH[j] = permutationH[ipick];
				permutationH[ipick] = temp;
			}

			for (j = 0; j < VN_degree[i]; j++)
			{
				ipick = unif_intH() % GF;

				if (ipick == 0)
				{
					j--;
					continue;
				}
				permutationH2[j] = ipick;
			}

			for (j = 0; j < VN_degree[i]; j++)
			{
				int CN = permutationH[j];
				VNsNeighbors[i][j] = CN;
				VNtoCNconnection[i][j] = permutationH2[j];

				CNtoVNconnection[CN][CN_degree_temp[CN]] = permutationH2[j];
				CNsNeighbors[CN][CN_degree_temp[CN]++] = i;
			}
		}
	}
	delete[] row;
	delete[] firstrow;
	delete[] interleaver;
}

///////////////////////////////////////////////////////
///Build parity check matrix of NB-LDPC code
///////////////////////////////////////////////////////

/*
void build_G(int dv, int dc, int n, int nk)
{
	int VN, CN, i, j, ii, jj;

	int temp[Def_n], index, index2;

	for (VN = nk; VN < n; VN++)
	{
		for (i = 0; i < n; i++)
			temp[i] = 0;

		for (CN = VN - nk; CN < n - nk; CN++)
		{
			if (Hinv[CN][VN] == 0)
				continue;
			else
			{
				for (i = 0; i < n; i++)
				{
					temp[i] = Hinv[CN][i];
					Hinv[CN][i] = Hinv[VN - nk][i];
					Hinv[VN - nk][i] = temp[i];
				}

				i = 0;
				while (i < n - nk)
				{
					if (Hinv[i][VN] == 0 || i == VN - nk)
					{
						i++;
						continue;
					}
					else
					{
						for (j = 1; j < GF; j++)
						{
							if (mult_table[j][Hinv[VN - nk][VN]] == Hinv[i][VN])
							{
								index2 = j;
								break;
							}
						}

						for (j = 0; j < n; j++)
						{
							if (Hinv[VN - nk][j] == 0)
								continue;
							else
								Hinv[i][j] = add_table[mult_table[index2][Hinv[VN - nk][j]]][Hinv[i][j]];
						}
					}
					i++;
				}

				for (i = 1; i < GF; i++)
				{
					if (mult_table[i][Hinv[VN - nk][VN]] == 1)
					{
						index2 = i;
						break;
					}
				}

				for (i = 0; i < n; i++)
				{
					if (Hinv[VN - nk][i] == 0)
						continue;
					else
					{
						Hinv[VN - nk][i] = mult_table[Hinv[VN - nk][i]][index2];
					}
				}
				break;
			}
		}
	}

	for (i = 0; i < nk; i++)
		G[i][i] = 1;

	for (; i < n; i++)
	{
		for (j = 0; j < nk; j++)
			G[j][i] = Hinv[i - nk][j];
	}
}
*/
/////////////////////////
/// Encoder
/////////////////////////
void encode(int* codeword, int n, int nk, int nw)
{
	int i, j;

	for (i = 0; i < n; i++)
		codeword_ref[i] = 0;

	/*
	for (i = 0; i < n; i++)
		for (j = 0; j < nk; j++)
			codeword_ref[i] = add_table[codeword_ref[i]][mult_table[u[j]][G[j][i]]];
	*/
	for (i = 0; i < n; i++)
	{
		offset_LDPC[i] = unif_int() % GF;
		//codeword[i] = codeword_ref[i];
		codeword[i] = offset_LDPC[i];
		codeword_ref[i] = 0;
	}

}
/////////////////////////
/// Encoder
/////////////////////////


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
	int sparsevect[Def_n * Def_nw] = { 0 };
	int sparsevect_DNA[Def_n * Def_nw / 2] = { 0 };

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
	int i, j;
	int ipick, temp;

	int i_trans = 0;
	int drift = 0;
	int ins = 0;
	int N = n * nw / 2;
	int rand;
	i_trans = 0;


	i_trans = 0;
	drift = 0;
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


	/*
	printf("drfit = %i\n", drift);
	//offset = 0;
	*/
	return(drift);
}
/////////////////////////
/// Add error vector
/////////////////////////



/////////////////////////
/// Watermark decoder
/////////////////////////
void watermark_decode(int recieved[][2 * Def_n * Def_nw / 2], int* drift, int n, int nw, int nk, int nkw, int* watermark, int* codebook, int rep_i)
{
	int i, j, ii, jj, edge, d_i, x_1, x_2;

	int I = Ins_max;

	int mid_point = drift_max;
	int n_edge = Codetrellis.nedge;
	int drift_state = 0;

	int rep = Rep_Factor;
	int rep_start, rep_tot;
	int count = 0;

	double sum, sum_norm, sum3, sum_norm2[Def_n] = { 0 };

	double Liklehoods_apriori[GF][Def_n];

	int N = n * nw;
	int dr_1[Rep_Factor], dr_2[Rep_Factor], length[Rep_Factor];
	int index[Rep_Factor], index_watermark, numb;
	int N_drift[Rep_Factor];
	int string[Def_nw] = { 0 };
	int string_DNA[Def_nw / 2] = { 0 };
	double Forward_point;
	double Backward_point;

	int max_length_rec = nw * (Ins_max + 1) + 1;

	int temp_recieved[Rep_Factor][20];
	int temp_transmitted[Def_nw / 2];
	int temp_received_int[Rep_Factor];
	int temp_transmitted_int;

	int decide_transm_int[GF][Def_n];

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

	for (i = 0; i < n; i++)
		for (j = 0; j < GF; j++)
			Liklehoods_apriori[j][i] = Liklehoods[rep_start][j][i];

	for (i = rep_start; i < rep_tot + rep_start; i++)
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

					Middle[jj][d_i - 1][i][j] = Receiver_metric(temp_recieved[jj], temp_transmitted, j, nw);
				}
			}
		}
	}

	for (i = 0; i < states_max; i++)
		for (j = 0; j < n + 1; j++)
			Forward[i][j] = 0;

	for (i = 0; i < states_max; i++)
		for (j = 0; j < 2; j++)
			Backward[i][j] = 0;

	if (rep_i == -1)
	{
		for (i = 0; i < states_max; i++)
		{
			if (state_mapper[i][0] == 0 && state_mapper[i][1] == 0)
			{
				mid_point = i;
				break;
			}
		}

		for (i = 0; i < states_max; i++)
		{
			if (state_mapper[i][0] == drift[0] && state_mapper[i][1] == drift[1])
			{
				drift_state = i;
				break;
			}
		}
	}
	else
	{
		mid_point = drift_max;
		drift_state = mid_point + drift[rep_start];
	}

	Forward[mid_point][0] = 1;
	Backward[drift_state][1] = 1;

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

			vec2int_q(temp_transmitted, temp_transmitted_int, nw, 4);

			decide_transm_int[i][d_i - 1] = temp_transmitted_int;
		}
	}

	//Forward recursion
	for (d_i = 1; d_i < n + 1; d_i++)
	{
		sum_norm = 0;

		for (edge = 0; edge < n_edge; edge++)
		{
			x_1 = Codetrellis.trellis[edge][0];
			for (i = rep_start; i < rep_tot + rep_start; i++)
				dr_1[i] = Codetrellis.trellis[edge][i - rep_start + 1];

			x_2 = Codetrellis.trellis[edge][rep_tot + 1];
			for (i = rep_start; i < rep_tot + rep_start; i++)
				dr_2[i] = Codetrellis.trellis[edge][rep_tot + 1 + i - rep_start + 1];

			count = 0;
			for (i = rep_start; i < rep_tot + rep_start; i++)
			{
				index[i] = (d_i - 1) * nw + dr_1[i];
				length[i] = nw + dr_2[i] - dr_1[i];

				if (index[i] < 0 || index[i] + length[i] > N_drift[i])
					count++;
			}

			if (count > 0)
				continue;

			for (i = 0; i < GF; i++)
			{
				temp_transmitted_int = decide_transm_int[i][d_i - 1];
				//int2vec_q(temp_transmitted, temp_transmitted_int, nw, 4);

				Forward_point = 1;
				for (ii = rep_start; ii < rep_tot + rep_start; ii++)
				{
					Forward_point *= Middle[ii][index[ii]][temp_transmitted_int][length[ii]];
					//double Forward_point2 = Receiver_metric(temp_recieved, temp_transmitted, length, nw);
				}

				Forward[x_2][d_i] += Forward[x_1][d_i - 1] * Forward_point * Liklehoods_apriori[i][d_i - 1];
			}//end loop over time t possible drift values
		}//end loop over time t - 1 possible drift values

		for (ii = 0; ii < states_max; ii++)
			sum_norm += Forward[ii][d_i];

		for (ii = 0; ii < states_max; ii++)
			Forward[ii][d_i] /= sum_norm;
	}//end loop over all transmitted bits

	for (i = 0; i < GF; i++)
		for (j = 0; j < n; j++)
			Liklehoods[rep_start][i][j] = 0;


	//Backward recursion
	for (d_i = n - 1; d_i >= 0; d_i--)
	{
		sum_norm = 0;

		for (edge = 0; edge < n_edge; edge++)
		{
			x_2 = Codetrellis.trellis[edge][0];
			for (i = rep_start; i < rep_tot + rep_start; i++)
				dr_2[i] = Codetrellis.trellis[edge][i - rep_start + 1];

			x_1 = Codetrellis.trellis[edge][rep_tot + 1];
			for (i = rep_start; i < rep_tot + rep_start; i++)
				dr_1[i] = Codetrellis.trellis[edge][rep_tot + 1 + i - rep_start + 1];

			count = 0;
			for (i = rep_start; i < rep_tot + rep_start; i++)
			{
				index[i] = d_i * nw + dr_2[i];
				length[i] = nw + dr_1[i] - dr_2[i];

				if (index[i] < 0 || index[i] + length[i] > N_drift[i])
					count++;
			}

			if (count > 0)
				continue;

			for (i = 0; i < GF; i++)
			{
				temp_transmitted_int = decide_transm_int[i][d_i];

				Backward_point = 1;
				for (ii = rep_start; ii < rep_tot + rep_start; ii++)
				{
					//int2vec_q(temp_transmitted, temp_transmitted_int, nw, 4);

					//Backward_point = Receiver_metric(temp_recieved, temp_transmitted, length, nw);
					Backward_point *= Middle[ii][index[ii]][temp_transmitted_int][length[ii]];
				}

				Backward[x_2][0] += Backward[x_1][1] * Backward_point * Liklehoods_apriori[i][d_i];

				Liklehoods[rep_start][i][d_i] += Backward_point * Forward[x_2][d_i] * Backward[x_1][1];
			}
		}//end loop over edges

		for (i = 0; i < GF; i++)
			sum_norm2[d_i] += Liklehoods[rep_start][i][d_i];

		for (ii = 0; ii < states_max; ii++)
			sum_norm += Backward[ii][0];

		for (ii = 0; ii < states_max; ii++)
		{
			Backward[ii][0] /= sum_norm;
			Backward[ii][1] = Backward[ii][0];
			Backward[ii][0] = 0;
		}
	}//end loop over all transmitted bits

	for (i = 0; i < GF; i++)
		for (j = 0; j < n; j++)
			Liklehoods[rep_start][i][j] /= sum_norm2[j];
	/*
		for (i = 0; i < n; i++)
		{
			sum_norm2[i] = 0;
			for (j = 0; j < GF; j++)
			{
				Liklehoods[rep_start][j][i] /= Liklehoods_apriori[j][i];
				sum_norm2[i] += Liklehoods[rep_start][j][i];
			}

		}

		for (i = 0; i < GF; i++)
			for (j = 0; j < n; j++)
				Liklehoods[rep_start][i][j] /= sum_norm2[j];
	*/
	double Liklehoods_temp[GF] = { 0 };

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < GF; j++)
		{
			Liklehoods_temp[j] = Liklehoods[rep_start][j][i];
		}
		for (j = 0; j < GF; j++)
		{
			Liklehoods[rep_start][add_table[j][offset_LDPC[i]]][i] = Liklehoods_temp[j];
		}
	}
}
/////////////////////////
/// Watermark decoder
/////////////////////////


/////////////////////////
/// Watermark decoder
/////////////////////////
void watermark_decode_test(int recieved[][2 * Def_n * Def_nw / 2], int* drift, int n, int nw, int nk, int nkw, int* watermark, int* codebook, int rep_i)
{
	int i, j, ii, jj, edge, d_i, x_1, x_2;

	int I = Ins_max;

	int mid_point = drift_max;
	int n_edge = Codetrellis.nedge;
	int drift_state = 0;

	int rep = Rep_Factor;
	int rep_start, rep_tot;
	int count = 0;

	double sum, sum_norm, sum3, sum_norm2[Def_n * Def_nw] = { 0 };

	int nw_base = 1;

	int N = n * nw;
	int dr_1[Rep_Factor], dr_2[Rep_Factor], length[Rep_Factor];
	int index[Rep_Factor], index_watermark, numb;
	int N_drift[Rep_Factor];
	int string[Def_nw] = { 0 };
	int string_DNA[Def_nw / 2] = { 0 };
	double Forward_point;
	double Backward_point;

	int temp_recieved[Rep_Factor][20];
	int temp_transmitted[Def_nw / 2];
	int temp_received_int[Rep_Factor];
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


	for (i = rep_start; i < rep_tot + rep_start; i++)
		N_drift[i] = N + drift[i];

	for (i = 0; i < states_max; i++)
		for (j = 0; j < N + 1; j++)
			Forward[i][j] = 0;

	for (i = 0; i < states_max; i++)
		for (j = 0; j < 2; j++)
			Backward[i][j] = 0;

	if (rep_i == -1)
	{
		for (i = 0; i < states_max; i++)
		{
			if (state_mapper[i][0] == 0 && state_mapper[i][1] == 0)
			{
				mid_point = i;
				break;
			}
		}

		for (i = 0; i < states_max; i++)
		{
			if (state_mapper[i][0] == drift[0] && state_mapper[i][1] == drift[1])
			{
				drift_state = i;
				break;
			}
		}
	}
	else
	{
		mid_point = drift_max;
		drift_state = mid_point + drift[rep_start];
	}

	Forward[mid_point][0] = 1;
	Backward[drift_state][1] = 1;


	//Forward recursion
	for (d_i = 1; d_i < N + 1; d_i++)
	{
		sum_norm = 0;

		for (edge = 0; edge < n_edge; edge++)
		{
			x_1 = Codetrellis.trellis[edge][0];
			for (i = rep_start; i < rep_tot + rep_start; i++)
				dr_1[i] = Codetrellis.trellis[edge][i - rep_start + 1];

			x_2 = Codetrellis.trellis[edge][rep_tot + 1];
			for (i = rep_start; i < rep_tot + rep_start; i++)
				dr_2[i] = Codetrellis.trellis[edge][rep_tot + 1 + i - rep_start + 1];

			count = 0;
			for (i = rep_start; i < rep_tot + rep_start; i++)
			{
				index[i] = (d_i - 1) * nw_base + dr_1[i];
				length[i] = nw_base + dr_2[i] - dr_1[i];

				if (index[i] < 0 || index[i] + length[i] > N_drift[i])
					count++;
			}

			if (count > 0)
				continue;

			for(i = rep_start; i < rep_tot + rep_start; i++)
				for(j = 0; j < length[i]; j++)
					temp_recieved[i][j] = recieved[i][index[i] + j];

			for (i = 0; i < 4; i++)
			{
				temp_transmitted[0] = i;

				Forward_point = 1;
				for (ii = rep_start; ii < rep_tot + rep_start; ii++)
				{
					Forward_point *= Receiver_metric(temp_recieved[ii], temp_transmitted, length[ii], nw_base);
					//double Forward_point2 = Receiver_metric(temp_recieved, temp_transmitted, length, nw);
				}

				Forward[x_2][d_i] += Forward[x_1][d_i - 1] * Forward_point * 0.25;
			}//end loop over time t possible drift values
		}//end loop over time t - 1 possible drift values

		for (ii = 0; ii < states_max; ii++)
			sum_norm += Forward[ii][d_i];

		for (ii = 0; ii < states_max; ii++)
			Forward[ii][d_i] /= sum_norm;
	}//end loop over all transmitted bits

	for (i = 0; i < GF; i++)
		for (j = 0; j < n; j++)
			Liklehoods[rep_start][i][j] = 1;


	//Backward recursion
	for (d_i = N - 1; d_i >= 0; d_i--)
	{
		sum_norm = 0;

		for (edge = 0; edge < n_edge; edge++)
		{
			x_2 = Codetrellis.trellis[edge][0];
			for (i = rep_start; i < rep_tot + rep_start; i++)
				dr_2[i] = Codetrellis.trellis[edge][i - rep_start + 1];

			x_1 = Codetrellis.trellis[edge][rep_tot + 1];
			for (i = rep_start; i < rep_tot + rep_start; i++)
				dr_1[i] = Codetrellis.trellis[edge][rep_tot + 1 + i - rep_start + 1];

			count = 0;
			for (i = rep_start; i < rep_tot + rep_start; i++)
			{
				index[i] = d_i * nw_base + dr_2[i];
				length[i] = nw_base + dr_1[i] - dr_2[i];

				if (index[i] < 0 || index[i] + length[i] > N_drift[i])
					count++;
			}

			if (count > 0)
				continue;

			for (i = rep_start; i < rep_tot + rep_start; i++)
				for (j = 0; j < length[i]; j++)
					temp_recieved[i][j] = recieved[i][index[i] + j];

			for (i = 0; i < 4; i++)
			{
				temp_transmitted[0] = i;

				Backward_point = 1;
				for (ii = rep_start; ii < rep_tot + rep_start; ii++)
				{
					Backward_point *= Receiver_metric(temp_recieved[ii], temp_transmitted, length[ii], nw_base);
					//double Forward_point2 = Receiver_metric(temp_recieved, temp_transmitted, length, nw);
				}

				Backward[x_2][0] += Backward[x_1][1] * Backward_point * 0.25;

				Liklehoods_test[rep_start][i][d_i] += Backward_point * Forward[x_2][d_i] * Backward[x_1][1];
			}
		}//end loop over edges

		for (i = 0; i < 4; i++)
			sum_norm2[d_i] += Liklehoods_test[rep_start][i][d_i];

		for (ii = 0; ii < states_max; ii++)
			sum_norm += Backward[ii][0];

		for (ii = 0; ii < states_max; ii++)
		{
			Backward[ii][0] /= sum_norm;
			Backward[ii][1] = Backward[ii][0];
			Backward[ii][0] = 0;
		}
	}//end loop over all transmitted bits

	for (i = 0; i < 4; i++)
		for (j = 0; j < N; j++)
			Liklehoods_test[rep_start][i][j] /= sum_norm2[j];
	/*
		for (i = 0; i < n; i++)
		{
			sum_norm2[i] = 0;
			for (j = 0; j < GF; j++)
			{
				Liklehoods[rep_start][j][i] /= Liklehoods_apriori[j][i];
				sum_norm2[i] += Liklehoods[rep_start][j][i];
			}

		}

		for (i = 0; i < GF; i++)
			for (j = 0; j < n; j++)
				Liklehoods[rep_start][i][j] /= sum_norm2[j];
	*/

	for (i = 0; i < n; i++)
	{
		sum_norm2[i] = 0;

		for (j = 0; j < GF; j++)
		{
			numb = LookUpSparse[codebook[i]][j];

			int2vec_q(string_DNA, numb, nw, 4);

			for (ii = 0; ii < nw; ii++)
				Liklehoods[rep_start][j][i] *= Liklehoods_test[rep_start][string_DNA[ii]][i * nw + ii];

			sum_norm2[i] += Liklehoods[rep_start][j][i];
		}
	}

	for (i = 0; i < GF; i++)
		for (j = 0; j < n; j++)
			Liklehoods[rep_start][i][j] /= sum_norm2[j];

	double Liklehoods_temp[GF] = { 0 };

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < GF; j++)
		{
			Liklehoods_temp[j] = Liklehoods[rep_start][j][i];
		}
		for (j = 0; j < GF; j++)
		{
			Liklehoods[rep_start][add_table[j][offset_LDPC[i]]][i] = Liklehoods_temp[j];
		}
	}
}
/////////////////////////
/// Watermark decoder
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

void normalize_log(double* x, int n)
{
	double sum = -INFINITY;
	for (int it = 0; it < n; it++) {
		if (x[it] != 0)
			sum = max_star(sum, x[it]);
	}
	for (int it = 0; it < n; it++) {
		if (x[it] != 0)
			x[it] = x[it] - sum;
	}
}


/////////////////////////
/// Watermark decoder
/////////////////////////
void watermark_decode_LogDomain(int recieved[][2 * Def_n * Def_nw / 2], int* drift, int n, int nw, int nk, int nkw, int* watermark, int* codebook, int rep_i)
{
	int i, j, ii, jj, edge, d_i, x_1, x_2;

	int I = Ins_max;

	int mid_point = drift_max;
	int n_edge = Codetrellis.nedge;
	int drift_state = 0;

	int rep = Rep_Factor;
	int rep_start, rep_tot;
	int count = 0;

	double sum, sum_norm, sum3, sum_norm2[Def_n] = { 0 };

	double Liklehoods_apriori[GF][Def_n];

	int N = n * nw;
	int dr_1[Rep_Factor], dr_2[Rep_Factor], length[Rep_Factor];
	int index[Rep_Factor], index_watermark, numb;
	int N_drift[Rep_Factor];
	int string[Def_nw] = { 0 };
	int string_DNA[Def_nw / 2] = { 0 };
	double Forward_point;
	double Backward_point;

	int max_length_rec = nw * (Ins_max + 1) + 1;

	int temp_recieved[Rep_Factor][20];
	int temp_transmitted[Def_nw / 2];
	int temp_received_int[Rep_Factor];
	int temp_transmitted_int;

	int decide_transm_int[GF][Def_n];

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

	for (i = 0; i < n; i++)
		for (j = 0; j < GF; j++)
			Liklehoods_apriori[j][i] = log(Liklehoods[rep_start][j][i]);

	for (i = rep_start; i < rep_tot + rep_start; i++)
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

	for (i = 0; i < states_max; i++)
		for (j = 0; j < 2; j++)
			Backward[i][j] = -INFINITY;

	if (rep_i == -1)
	{
		for (i = 0; i < states_max; i++)
		{
			if (state_mapper[i][0] == 0 && state_mapper[i][1] == 0)
			{
				mid_point = i;
				break;
			}
		}

		for (i = 0; i < states_max; i++)
		{
			if (state_mapper[i][0] == drift[0] && state_mapper[i][1] == drift[1])
			{
				drift_state = i;
				break;
			}
		}
	}
	else
	{
		mid_point = drift_max;
		drift_state = mid_point + drift[rep_start];
	}

	Forward[mid_point][0] = 0;
	Backward[drift_state][1] = 0;

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

			vec2int_q(temp_transmitted, temp_transmitted_int, nw, 4);

			decide_transm_int[i][d_i - 1] = temp_transmitted_int;
		}
	}

	//Forward recursion
	for (d_i = 1; d_i < n + 1; d_i++)
	{
		sum_norm = 0;

		for (edge = 0; edge < n_edge; edge++)
		{
			x_1 = Codetrellis.trellis[edge][0];
			for (i = rep_start; i < rep_tot + rep_start; i++)
				dr_1[i] = Codetrellis.trellis[edge][i - rep_start + 1];

			x_2 = Codetrellis.trellis[edge][rep_tot + 1];
			for (i = rep_start; i < rep_tot + rep_start; i++)
				dr_2[i] = Codetrellis.trellis[edge][rep_tot + 1 + i - rep_start + 1];

			count = 0;
			for (i = rep_start; i < rep_tot + rep_start; i++)
			{
				index[i] = (d_i - 1) * nw + dr_1[i];
				length[i] = nw + dr_2[i] - dr_1[i];

				if (index[i] < 0 || index[i] + length[i] > N_drift[i])
					count++;
			}

			if (count > 0)
				continue;

			for (i = 0; i < GF; i++)
			{
				temp_transmitted_int = decide_transm_int[i][d_i - 1];
				//int2vec_q(temp_transmitted, temp_transmitted_int, nw, 4);

				Forward_point = 0;
				for (ii = rep_start; ii < rep_tot + rep_start; ii++)
				{
					Forward_point += Middle[ii][index[ii]][temp_transmitted_int][length[ii]];
					//double Forward_point2 = Receiver_metric(temp_recieved, temp_transmitted, length, nw);
				}

				Forward[x_2][d_i] = max_starLU(Forward[x_2][d_i], Forward[x_1][d_i - 1] + Forward_point + Liklehoods_apriori[i][d_i - 1]);
			}//end loop over time t possible drift values
		}//end loop over time t - 1 possible drift values
	}//end loop over all transmitted bits


	for (i = 0; i < GF; i++)
		for (j = 0; j < n; j++)
			Liklehoods[rep_start][i][j] = -INFINITY;

	//Backward recursion
	for (d_i = n - 1; d_i >= 0; d_i--)
	{
		sum_norm = 0;

		for (edge = 0; edge < n_edge; edge++)
		{
			x_2 = Codetrellis.trellis[edge][0];
			for (i = rep_start; i < rep_tot + rep_start; i++)
				dr_2[i] = Codetrellis.trellis[edge][i - rep_start + 1];

			x_1 = Codetrellis.trellis[edge][rep_tot + 1];
			for (i = rep_start; i < rep_tot + rep_start; i++)
				dr_1[i] = Codetrellis.trellis[edge][rep_tot + 1 + i - rep_start + 1];

			count = 0;
			for (i = rep_start; i < rep_tot + rep_start; i++)
			{
				index[i] = d_i * nw + dr_2[i];
				length[i] = nw + dr_1[i] - dr_2[i];

				if (index[i] < 0 || index[i] + length[i] > N_drift[i])
					count++;
			}

			if (count > 0)
				continue;

			for (i = 0; i < GF; i++)
			{
				temp_transmitted_int = decide_transm_int[i][d_i];

				Backward_point = 0;
				for (ii = rep_start; ii < rep_tot + rep_start; ii++)
				{
					//int2vec_q(temp_transmitted, temp_transmitted_int, nw, 4);

					//Backward_point = Receiver_metric(temp_recieved, temp_transmitted, length, nw);
					Backward_point += Middle[ii][index[ii]][temp_transmitted_int][length[ii]];
				}

				Backward[x_2][0] = max_starLU(Backward[x_2][0], Backward[x_1][1] + Backward_point + Liklehoods_apriori[i][d_i]);

				Liklehoods[rep_start][i][d_i] = max_starLU(Liklehoods[rep_start][i][d_i], Backward_point + Forward[x_2][d_i] + Backward[x_1][1]);
			}
		}//end loop over edges


		for (ii = 0; ii < states_max; ii++)
		{
			Backward[ii][1] = Backward[ii][0];
			Backward[ii][0] = -INFINITY;
		}
	}//end loop over all transmitted bits


	double Liklehoods_temp[GF] = { 0 };

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < GF; j++)
		{
			Liklehoods_temp[j] = Liklehoods[rep_i][j][i];
		}
		for (j = 0; j < GF; j++)
		{
			Liklehoods[rep_i][add_table[j][offset_LDPC[i]]][i] = Liklehoods_temp[j];
		}
	}

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < GF; j++)
		{
			Liklehoods_temp[j] = Liklehoods[rep_i][j][i];
		}

		normalize_log(Liklehoods_temp, GF);

		for (j = 0; j < GF; j++)
		{
			Liklehoods[rep_i][j][i] = exp(Liklehoods_temp[j]);
		}
	}
}
/////////////////////////
/// Watermark decoder
/////////////////////////



/////////////////////////
/// NTT calculation
/////////////////////////
void ntt(double* p) // Multi-dimensional Fourier Transform
{
	int factor = 1;
	for (int b = 0; b < log2(GF); b++)
	{
		for (int rest = 0; rest < GF / 2; rest++)
		{
			int restH = rest >> b;
			int restL = rest & (factor - 1);
			int rest0 = (restH << (b + 1)) + restL;
			int rest1 = rest0 + factor;
			double prest0 = p[rest0];
			p[rest0] += p[rest1];
			p[rest1] = prest0 - p[rest1];
		}
		factor += factor;
	}
}
/////////////////////////
/// NTT calculation
/////////////////////////


/////////////////////////
/// Parallel Decoder
/////////////////////////
void parallel_decoding(int n, int rep_factor)
{
	int i, j, ii;
	double prod[GF][Def_n];
	double sum_norm[Def_n] = { 0 };

	for (i = 0; i < n; i++)
		for (j = 0; j < GF; j++)
			prod[j][i] = 1;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < GF; j++)
		{
			for (ii = 0; ii < rep_factor; ii++)
				prod[j][i] *= Liklehoods[ii][j][i];

			sum_norm[i] += prod[j][i];
		}
	}

	for (i = 0; i < GF; i++)
		for (j = 0; j < n; j++)
			prod[i][j] /= sum_norm[j];

	for (i = 0; i < n; i++)
		for (j = 0; j < GF; j++)
			Liklehoods[0][j][i] = prod[j][i];

}
/////////////////////////
/// Parallel Decoder
/////////////////////////



/////////////////////////
/// NB-LDPC decoder (SPA)
/////////////////////////
bool decode_SPA(int* decoded, int dv, int dc, int n, int nk)
{
	int i, j, ii, jj;
	int iter;

	int VN;
	int CN;

	double prod[Def_n][GF];
	double prod2[GF];
	double prod_fft[GF];
	int Neighbor[30] = { 0 };

	int symberrs; //symbol errors in the decoded codeword

	double temp1, temp2;
	double f_sum;
	double div;
	double norm_sum;
	double sum_norm[Def_n];

	double x[Def_dc + 1][GF];
	double y[GF];

	int rep = Rep_Factor;

	bool exit = 0;

	for (VN = 0; VN < n; VN++)
		for (i = 0; i < VN_degree[VN]; i++)
			for (j = 0; j < GF; j++)
				mVNtoCN[VN][i][j] = Liklehoods[0][j][VN];

	for (iter = 0; iter < MaxNumIt; iter++)
	{
		//check node update
		for (CN = 0; CN < nk; CN++)
		{
			int isnegative[Def_dc][GF];
			int sgnsum[GF] = { 0 };

			//int* CN_Neighbs = CNsNeighbors[CN];
			//int* CNtoVN_connect = CNtoVNconnection[CN];

			for (jj = 0; jj < GF; jj++)
				prod_fft[jj] = 1;

			for (j = 0; j < CN_degree[CN]; j++)
			{
				int* mult = mult_table[CNtoVNconnection[CN][j]];

				VN = CNsNeighbors[CN][j];

				for (jj = 0; jj < VN_degree[VN]; jj++)
				{
					if (VNsNeighbors[VN][jj] == CN)
						Neighbor[0] = jj;
				}

				for (jj = 0; jj < GF; jj++)
				{
					x[j][mult[jj]] = mVNtoCN[VN][Neighbor[0]][jj];
				}

				ntt(x[j]);

				for (jj = 0; jj < GF; jj++)
				{
					if (x[j][jj] < 0)
					{
						x[j][jj] = -x[j][jj];
						isnegative[j][jj] = 1;
						sgnsum[jj] ^= 1;
					}
					else
						isnegative[j][jj] = 0;
				}

			}//end for loop over CN neighbors

			for (j = 0; j < CN_degree[CN]; j++)
				for (jj = 0; jj < GF; jj++)
					prod_fft[jj] *= x[j][jj];

			for (j = 0; j < CN_degree[CN]; j++)
			{
				int* mult = mult_table[CNtoVNconnection[CN][j]];

				for (jj = 0; jj < GF; jj++)
					y[jj] = prod_fft[jj] / x[j][jj];

				for (jj = 0; jj < GF; jj++)
				{
					if (isnegative[j][jj] != sgnsum[jj])
						y[jj] = -y[jj];
				}//end for loop over GF symbols 

				ntt(y);

				norm_sum = 0;

				for (jj = 0; jj < GF; jj++)
				{
					if (y[jj] == 0 || y[jj] < 0)
						y[jj] = 1e-48;

					if (y[jj] == GF)
						y[jj] -= 1e-48;

					norm_sum += y[jj];
				}//end for loop over GF symbols 

				for (jj = 0; jj < GF; jj++)
					y[jj] /= norm_sum;

				for (jj = 0; jj < GF; jj++)
					mCNtoVN[CN][j][jj] = y[mult[jj]];


			}//end for loop over CN neighbors
		}//end for loop over CNs

		//end check node update

		for (VN = 0; VN < n; VN++)
			for (i = 0; i < GF; i++)
				prod[VN][i] = 1;

		//variable node update

		for (VN = 0; VN < n; VN++)
		{
			sum_norm[VN] = 0;
			for (ii = 0; ii < GF; ii++)
			{
				for (i = 0; i < VN_degree[VN]; i++)
				{
					CN = VNsNeighbors[VN][i];

					for (jj = 0; jj < CN_degree[CN]; jj++)
					{
						if (CNsNeighbors[CN][jj] == VN)
							Neighbor[i] = jj;
					}

					prod[VN][ii] *= mCNtoVN[CN][Neighbor[i]][ii];
				}//end for loop over VN neighbors

				prod[VN][ii] *= Liklehoods[0][ii][VN];
				sum_norm[VN] += prod[VN][ii];
			}//end for loop over GF symbols 

			for (ii = 0; ii < GF; ii++)
				prod2[ii] = 1;

			for (i = 0; i < VN_degree[VN]; i++)
			{
				f_sum = 0;
				CN = VNsNeighbors[VN][i];
				for (ii = 0; ii < GF; ii++)
				{
					prod2[ii] = prod[VN][ii];
					prod2[ii] /= mCNtoVN[CN][Neighbor[i]][ii];
					f_sum += prod2[ii];
				}//end for loop over VN neighbors

				for (ii = 0; ii < GF; ii++)
				{
					mVNtoCN[VN][i][ii] = prod2[ii] / f_sum;
				}//end for loop over GF symbols
			}//end for loop over GF symbols 

		}//end for loop over VNs
		//end check node update


		//symbol Decsicion
		symberrs = 0;

		for (VN = 0; VN < n; VN++)
		{
			int max = unif_int() % GF;
			double max_prod = 0;

			for (i = 0; i < GF; i++)
			{
				if (prod[VN][i] > max_prod)
				{
					max_prod = prod[VN][i];
					max = i;
				}
				else
					continue;
			}//end for loop over GF symbols

			decoded[VN] = max;


			if (decoded[VN] != codeword_ref[VN])
			{
				symberrs++;
			}


		}//end for loop over VNs

		//printf("err = %i ",symberrs);
		//end symbol Decsicion

		if (symberrs == 0)
		{
			exit = 1;
			break;
		}
	} //for loop over iterations
	//printf("Num of iterations = %d\n", iter);

	for (VN = 0; VN < n; VN++)
		for (i = 0; i < GF; i++)
			prod[VN][i] /= sum_norm[VN];

	//Provide extrinsic information for iterative (turbo) decoding
	for (VN = 0; VN < n; VN++)
	{
		sum_norm[VN] = 0;

		for (i = 0; i < GF; i++)
		{
			Liklehoods[0][i][VN] = prod[VN][i] / Liklehoods[0][i][VN];
			sum_norm[VN] += Liklehoods[0][i][VN];
		}
	}

	for (i = 0; i < GF; i++)
		for (j = 0; j < n; j++)
			Liklehoods[0][i][j] /= sum_norm[j];


	double Liklehoods_temp[GF] = { 0 };

	//Adjust likelihoods according to the LDPC offset
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < GF; j++)
		{
			Liklehoods_temp[j] = Liklehoods[0][j][i];
		}
		for (j = 0; j < GF; j++)
		{
			if (Joint_Decoding == 0)
				for (ii = 0; ii < rep; ii++)
					Liklehoods[ii][add_table[j][offset_LDPC[i]]][i] = Liklehoods_temp[j];
			else
				Liklehoods[0][add_table[j][offset_LDPC[i]]][i] = Liklehoods_temp[j];
		}
	}

	return(exit);

}
/////////////////////////
/// NB-LDPC decoder (SPA)
/////////////////////////



void errorrate_computation(int* codeword, int* decoded, int n)
{
	int i, j;
	int numbiterrors = 0, numsymberrors = 0;

	for (i = 0; i < n; i++)
	{
		int temp;
		temp = add_table[add_table[codeword[i]][offset_LDPC[i]]][decoded[i]];

		if (temp > 0)
			numsymberrors++;

		for (j = 0; j < log2(GF); j++)
			numbiterrors += (temp >> j) & 1LL;
	}

	if (numbiterrors > 0)
	{
		bit_err += numbiterrors;
		symb_err += numsymberrors;
		frame_err++;
	}
}


int main(int argc, char** argv)
{
	int i, j, ii;

	int    n = Def_n;        // Codeword length
	int    nw = Def_nw;
	int    nk = Def_nk;       // Number of check nodes
	int    nkw = Def_nkw;       // Number of check nodes
	int    dv = Def_dv;       // Variable node degree
	int    dc = Def_dc;       // Check node degree
	int    Bk = Def_Bk;
	int    Bn = Def_Bn;
	int    rep = Rep_Factor;  //Repetition order
	int    turboIterations = Turbo_Iterations;

	int    watermark[Def_n * Def_nw / 2] = { 0 };
	int    receivedword[Rep_Factor][2 * Def_n * Def_nw / 2];
	int    codebook[Def_n] = { 0 };
	int    codeword[Def_n];         //NB-LDPC codeword
	int    codeword_water[Def_n * Def_nw / 2];   //Translated NB-LDPC codeword + watermark vector//Transmitted vector + error vector
	int    codeworddec[Def_n];      //decoded codeword
	int    drift[Rep_Factor];                   //# of insertions - # of deletions after transmittion

	int sim;
	int num_points = Def_NUM_POINTS;

	if (output_type == 1)
	{
		l_orig = Def_lstart;
		l_start_orig = Def_lstart;
		vc_un_orig = atoi(argv[1]) * (atoi(argv[7]) + 1);
		vc_unInter_orig = atoi(argv[2]) * (atoi(argv[7]) + 1);
		vc_unInterH_orig = atoi(argv[3]) * (atoi(argv[7]) + 1);
		vc_unch_orig = atoi(argv[4]) * (atoi(argv[7]) + 1);
		numb_frame_err = atoi(argv[5]);
		num_points = atoi(argv[6]);
	}
	else
	{
		l_orig = Def_lstart;
		l_start_orig = Def_lstart;
		vc_un_orig = Def_vcunstart;
		vc_unInter_orig = Def_vc_unInter;
		vc_unInterH_orig = Def_vc_unInterH;
		vc_unch_orig = 2000;
		numb_frame_err = Def_MinNumberFramesError;
		num_points = Def_NUM_POINTS;
	}

	initialize_variables(n);

	if (LDPC_build_type > 0)
	{
		//build_paritycheckmatrix
		build_paritycheckmatrix_v02(n, nk, Bk, Bn, dv, dc, LDPC_build_type, 0);
	}
	for (sim = 0; sim < num_points; sim++)
	{ // loop on simulation points
		initialize_sim(n, nw, sim);
		/*
		FILE* fout;

		char stringa[500];

		sprintf(stringa, "XY_vectors_Pid=%f.txt", P_ins);

		fout = fopen(stringa, "w");
		*/
		build_inner_codebook(nw, inner_codebook_design);
		for (frame = 0; frame < 4000; frame++)
		{

			//build_paritycheckmatrix
			build_paritycheckmatrix_v02(n, nk, Bk, Bn, dv, dc, LDPC_build_type, sim);

			//build generator matrix
			//build_G(dv, dc, n, nk);

			//Encoding
			encode(codeword, n, nk, nw);

			//Translate and add watermark 
			encode_water(codeword, codeword_water, n, nw, watermark, codebook, added_sequence_type);

			while (true)
			{
				for (i = 0; i < rep; i++)
				{
					//adding error vector
					drift[i] = add_error_vector(codeword_water, receivedword[i], n, nw);
				}

				if (drift[0] == 0)
					break;
			}
			/*
			int vect[2 * Def_n] = { 0 };

			for (i = 0; i < n; i++)
				vect[i] = codeword_water[i];

			for (i = n; i < 2 * n; i++)
				vect[i] = receivedword[0][i - n];

			for (i = 0; i < 2 * n; i++)
				fprintf(fout, "%i ", vect[i]);

			fprintf(fout, "\n");
			*/
			
			for (ii = 0; ii < rep; ii++)
				for (i = 0; i < n; i++)
					for (j = 0; j < GF; j++)
						Liklehoods[ii][j][i] = double(1) / GF;

			for (ii = 0; ii < turboIterations; ii++)
			{
				if (inner_decoder_domain == 0)
				{
					if (Joint_Decoding == 0)
					{
						for (i = 0; i < rep; i++)
						{
							//watermark decoder
							watermark_decode(receivedword, drift, n, nw / 2, nk, nkw, watermark, codebook, i);

							//watermark_decode_test(receivedword, drift, n, nw / 2, nk, nkw, watermark, codebook, i);
						}
					}
					else
						watermark_decode(receivedword, drift, n, nw / 2, nk, nkw, watermark, codebook, -1);
				}
				else
				{
					if (Joint_Decoding == 0)
					{
						for (i = 0; i < rep; i++)
						{
							//watermark decoder
							watermark_decode_LogDomain(receivedword, drift, n, nw / 2, nk, nkw, watermark, codebook, i);
						}
					}
					else
						watermark_decode_LogDomain(receivedword, drift, n, nw / 2, nk, nkw, watermark, codebook, -1);
				}

				if (Joint_Decoding == 0)
				{
					//Cobine LLRs before feeding to outer code
					parallel_decoding(n, rep);
				}
				//NB-LDPC decoder
				if (decode_SPA(codeworddec, dv, dc, n, nk))
					break;
			}
			//error rate computation
			errorrate_computation(codeword, codeworddec, n);

			if (willIstop())
				break;

			if (output_type == 0)
			{
				if ((frame + 1) % ncheck == 0)
					print_intermediate_results(n, P_ins, P_del, P_subs, sim);
			}
			else
				continue;
			
		}//END loop on frames

		//fclose(fout);

		//print results for the given simulated point
		if (output_type == 0)
		{
			printresults(sim, n, nk, nw, nkw, dv, dc, P_ins, P_del, P_subs, 0);
		}
		else
			printresults(sim, n, nk, nw, nkw, dv, dc, P_ins, P_del, P_subs, atoi(argv[7]));

	} // END loop on simulation points
	return 1;
}