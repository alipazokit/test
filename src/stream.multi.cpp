
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector> 
//#include <random>

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/QR>
#include "time.h"

#include "genotype.h"
#include "mailman.h"
#include "arguments.h"
//#include "helper.h"
#include "storage.h"

#include "/usr/include/boost/random.hpp"

#if SSE_SUPPORT==1
	#define fastmultiply fastmultiply_sse
	#define fastmultiply_pre fastmultiply_pre_sse
#else
	#define fastmultiply fastmultiply_normal
	#define fastmultiply_pre fastmultiply_pre_normal
#endif

using namespace Eigen;
using namespace std;

// Storing in RowMajor Form
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdr;
//typedef Matrix<int, Dynamic, Dynamic, RowMajor> MatrixXdrInt;

class data {

 public:
     MatrixXdr gen;
     int index;

};


//Intermediate Variables
int blocksize;
int hsegsize;
double *partialsums;
double *sum_op;		
double *yint_e;
double *yint_m;
double **y_e;
double **y_m;


struct timespec t0;

clock_t total_begin = clock();
MatrixXdr pheno;
MatrixXdr mask;
MatrixXdr covariate;  
MatrixXdr Q;
MatrixXdr v1; //W^ty
MatrixXdr v2;            //QW^ty
MatrixXdr v3;    //WQW^ty
MatrixXdr new_pheno;



genotype g;
genotype g1;
genotype g2;
MatrixXdr geno_matrix; //(p,n)
genotype* Geno;
int MAX_ITER;
int k,p,n;
int k_orig;

MatrixXdr c; //(p,k)
MatrixXdr x; //(k,n)
MatrixXdr v; //(p,k)
MatrixXdr means; //(p,1)
MatrixXdr stds; //(p,1)
MatrixXdr sum2;
MatrixXdr sum;  

Eigen::RowVectorXd means_na;
Eigen::RowVectorXd stds_na;
////////
//related to phenotype	
double y_sum; 
double y_mean;

options command_line_opts;

bool debug = false;
bool check_accuracy = false;
bool var_normalize=false;
int accelerated_em=0;
double convergence_limit;
bool memory_efficient = false;
bool missing=false;
bool fast_mode = true;
bool text_version = false;
bool use_cov=false; 


//// jackknife index wich are computed based on annotation file
MatrixXdr dic_index;
MatrixXdr jack_bin_size;
vector<int> len;
vector<int> Annot;
int Njack=100;
int Nbin=160;
int Nz=10;
///////

//define random vector z's
MatrixXdr  all_zb;
MatrixXdr  all_Uzb;
MatrixXdr res;
MatrixXdr XXz;
MatrixXdr Xy;
MatrixXdr yXXy;


///
//Matrix<int, Dynamic, Dynamic, RowMajor> gen;
MatrixXdr gen;
bool read_header;
//read variables
unsigned char mask2;
int wordsize;
unsigned int unitsperword;
int unitsize;
int nrow, ncol;
unsigned char *gtype;
int Nsnp;
int Nindv;
bool **bin_annot;
int step_size;
int step_size_rem;
std::vector<std::vector<bool> > annot_bool;
std::vector<std::vector<int> > jack_bin;
vector <data> allgen;
vector <genotype> allgen_mail;
int global_snp_index;
bool use_mailman=true;

///reading single col annot
vector <int> SNP_annot;
bool use_1col_annot=false;


///Variables for reg out cov on both side of LM
bool both_side_cov=true;
MatrixXdr UXXz;
MatrixXdr XXUz;
MatrixXdr Xz;
MatrixXdr trVK;

std::istream& newline(std::istream& in)
{
    if ((in >> std::ws).peek() != std::char_traits<char>::to_int_type('\n')) {
        in.setstate(std::ios_base::failbit);
    }
    return in.ignore();
}







void multiply_y_pre_fast(MatrixXdr &op, int Ncol_op ,MatrixXdr &res,bool subtract_means){

        for(int k_iter=0;k_iter<Ncol_op;k_iter++){
                sum_op[k_iter]=op.col(k_iter).sum();
        }

                        //cout << "Nops = " << Ncol_op << "\t" <<g.Nsegments_hori << endl;
        #if DEBUG==1
                if(debug){
                        print_time ();
                        cout <<"Starting mailman on premultiply"<<endl;
                        cout << "Nops = " << Ncol_op << "\t" <<g.Nsegments_hori << endl;
                        cout << "Segment size = " << g.segment_size_hori << endl;
                        cout << "Matrix size = " <<g.segment_size_hori<<"\t" <<g.Nindv << endl;
                        cout << "op = " <<  op.rows () << "\t" << op.cols () << endl;
                }
        #endif


        //TODO: Memory Effecient SSE FastMultipy

        for(int seg_iter=0;seg_iter<g.Nsegments_hori-1;seg_iter++){
                mailman::fastmultiply(g.segment_size_hori,g.Nindv,Ncol_op,g.p[seg_iter],op,yint_m,partialsums,y_m);
                int p_base = seg_iter*g.segment_size_hori;
                for(int p_iter=p_base; (p_iter<p_base+g.segment_size_hori) && (p_iter<g.Nsnp) ; p_iter++ ){
                        for(int k_iter=0;k_iter<Ncol_op;k_iter++)
                                res(p_iter,k_iter) = y_m[p_iter-p_base][k_iter];
                }
        }

        int last_seg_size = (g.Nsnp%g.segment_size_hori !=0 ) ? g.Nsnp%g.segment_size_hori : g.segment_size_hori;
        mailman::fastmultiply(last_seg_size,g.Nindv,Ncol_op,g.p[g.Nsegments_hori-1],op,yint_m,partialsums,y_m);
        int p_base = (g.Nsegments_hori-1)*g.segment_size_hori;
        for(int p_iter=p_base; (p_iter<p_base+g.segment_size_hori) && (p_iter<g.Nsnp) ; p_iter++){
                for(int k_iter=0;k_iter<Ncol_op;k_iter++)
                        res(p_iter,k_iter) = y_m[p_iter-p_base][k_iter];
        }

        #if DEBUG==1
                if(debug){
                        print_time ();
                        cout <<"Ending mailman on premultiply"<<endl;
                }
        #endif


        if(!subtract_means)
                return;

        for(int p_iter=0;p_iter<p;p_iter++){
                for(int k_iter=0;k_iter<Ncol_op;k_iter++){
                        res(p_iter,k_iter) = res(p_iter,k_iter) - (g.get_col_mean(p_iter)*sum_op[k_iter]);
                        if(var_normalize)
                                res(p_iter,k_iter) = res(p_iter,k_iter)/(g.get_col_std(p_iter));
                }
        }

}



void multiply_y_post_fast(MatrixXdr &op_orig, int Nrows_op, MatrixXdr &res,bool subtract_means){

        MatrixXdr op;
        op = op_orig.transpose();

        if(var_normalize && subtract_means){
                for(int p_iter=0;p_iter<p;p_iter++){
                        for(int k_iter=0;k_iter<Nrows_op;k_iter++)
                                op(p_iter,k_iter) = op(p_iter,k_iter) / (g.get_col_std(p_iter));
                }
        }

        #if DEBUG==1
                if(debug){
                        print_time ();
                        cout <<"Starting mailman on postmultiply"<<endl;
                }
        #endif

        int Ncol_op = Nrows_op;

        //cout << "ncol_op = " << Ncol_op << endl;

        int seg_iter;
        for(seg_iter=0;seg_iter<g.Nsegments_hori-1;seg_iter++){
mailman::fastmultiply_pre(g.segment_size_hori,g.Nindv,Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter],op,yint_e,partialsums,y_e);
        }
        int last_seg_size = (g.Nsnp%g.segment_size_hori !=0 ) ? g.Nsnp%g.segment_size_hori : g.segment_size_hori;
        mailman::fastmultiply_pre(last_seg_size,g.Nindv,Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter],op,yint_e,partialsums,y_e);

        for(int n_iter=0; n_iter<n; n_iter++)  {
                for(int k_iter=0;k_iter<Ncol_op;k_iter++) {
                        res(k_iter,n_iter) = y_e[n_iter][k_iter];
                        y_e[n_iter][k_iter] = 0;
                }
        }

        #if DEBUG==1
                if(debug){
                        print_time ();
                        cout <<"Ending mailman on postmultiply"<<endl;
                }
        #endif


        if(!subtract_means)
                return;

        double *sums_elements = new double[Ncol_op];
        memset (sums_elements, 0, Nrows_op * sizeof(int));

        for(int k_iter=0;k_iter<Ncol_op;k_iter++){
                double sum_to_calc=0.0;
                for(int p_iter=0;p_iter<p;p_iter++)
                        sum_to_calc += g.get_col_mean(p_iter)*op(p_iter,k_iter);
                sums_elements[k_iter] = sum_to_calc;
        }
        for(int k_iter=0;k_iter<Ncol_op;k_iter++){
                for(int n_iter=0;n_iter<n;n_iter++)
                        res(k_iter,n_iter) = res(k_iter,n_iter) - sums_elements[k_iter];
        }


}


void initial_var()
{
    /*if(key==1)
        g=g1;
    if(key==2)
        g=g2;*/
   // g=Geno[key];
        

	p = g.Nsnp;
        n = g.Nindv;


        c.resize(p,k);
        x.resize(k,n);
        v.resize(p,k);
        //means.resize(p,1);
        //stds.resize(p,1);
        sum2.resize(p,1);
        sum.resize(p,1);


        if(!fast_mode && !memory_efficient){
                geno_matrix.resize(p,n);
                g.generate_eigen_geno(geno_matrix,var_normalize);
        }

        //TODO: Initialization of c with gaussian distribution
        c = MatrixXdr::Random(p,k);


        // Initial intermediate data structures
        blocksize = k;
         hsegsize = g.segment_size_hori;        // = log_3(n)
        int hsize = pow(3,hsegsize);
        int vsegsize = g.segment_size_ver;              // = log_3(p)
        int vsize = pow(3,vsegsize);

        partialsums = new double [blocksize];
        sum_op = new double[blocksize];
        yint_e = new double [hsize*blocksize];
        yint_m = new double [hsize*blocksize];
        memset (yint_m, 0, hsize*blocksize * sizeof(double));
        memset (yint_e, 0, hsize*blocksize * sizeof(double));

        y_e  = new double*[g.Nindv];
        for (int i = 0 ; i < g.Nindv ; i++) {
                y_e[i] = new double[blocksize];
                memset (y_e[i], 0, blocksize * sizeof(double));
        }

        y_m = new double*[hsegsize];
        for (int i = 0 ; i < hsegsize ; i++)
                y_m[i] = new double[blocksize];
      /*  for(int i=0;i<p;i++){
                means(i,0) = g.get_col_mean(i);
                stds(i,0) =1/g.get_col_std(i);
                //sum2(i,0) =g.get_col_sum2(i); 
                sum(i,0)= g.get_col_sum(i);
        }

*/


}
	
MatrixXdr  compute_Xtz (int num_snp,MatrixXdr vec,int numcol){
	
         res.resize(num_snp, numcol);

        
	multiply_y_pre_fast(vec,numcol,res, false);
   

        MatrixXdr zb_sum = vec.colwise().sum();
        

	for(int j=0; j<num_snp; j++)
            for(int k=0; k<numcol;k++)
                res(j,k) = res(j,k)*stds(j,0);
            
        MatrixXdr resid(num_snp, numcol);
        MatrixXdr inter = means.cwiseProduct(stds);
        resid = inter * zb_sum;
        MatrixXdr inter_zb = res - resid;
       


	return inter_zb;
       

}



MatrixXdr  compute_Xz (int num_snp,MatrixXdr vec, int numcol){


	 MatrixXdr new_res(numcol, Nindv);         

                multiply_y_post_fast(vec,numcol,new_res, false);


        MatrixXdr new_resid(numcol, num_snp);
       MatrixXdr zb_scale_sum = vec * means;
       new_resid = zb_scale_sum * MatrixXdr::Constant(1,Nindv, 1);


                      /// new zb 
       MatrixXdr temp=new_res - new_resid;





	return temp.transpose();
}








void read_annot (string filename){
         
//	int step_size=Nsnp/Njack;
  //      int step_size_rem=Nsnp%Njack;
        vector<bool> snp_annot;
	//jack_bin.resize(Njack, vector<int>(Nbin,0));	
	
//	cout<<step_size<<endl;

	ifstream inp(filename.c_str());
        if (!inp.is_open()){
                cerr << "Error reading file "<< filename <<endl;
                exit(1);
        }
        string line;
        int j = 0 ;
        int linenum = 0 ;
        int num_parti;
        stringstream check1(line);
        string intermediate;
        vector <string> tokens;
        while(std::getline (inp, line)){
                //linenum ++;
                char c = line[0];
                if (c=='#')
                        continue;
                istringstream ss (line);
                if (line.empty())
                        continue;
                j++;
                //cout<<line<<endl;

                stringstream check1(line);
                string intermediate;
                vector <string> tokens;
                // Tokenizing w.r.t. space ' ' 
                while(getline(check1, intermediate, ' '))
                 {
                      tokens.push_back(intermediate);
                 }
                 if(linenum==0){
                 num_parti=tokens.size();
		 Nbin=num_parti;
                  snp_annot.resize(Nbin,0);	 
          	  jack_bin.resize(Njack, vector<int>(Nbin,0));
	         len.resize(num_parti,0);
                }
                int index_annot=0;
                for(int i = 0; i < tokens.size(); i++){
		      snp_annot[i]=0;
		      if (tokens[i]=="1"){
                            len[i]++;
			    snp_annot[i]=1;
 		      }
                }
		annot_bool.push_back(snp_annot);
        linenum++;
       }

	  if(Nsnp!=linenum){
          cout<<"Number of the rows in bim file and annotation file does not match"<<endl;
        }

	Nsnp=linenum;
	//cout<<"Total number of SNPs : "<<Nsnp<<endl;
	int selected_snps=0;
	for (int i=0;i<num_parti;i++){
                cout<<len[i]<<" SNPs in "<<i<<"-th bin"<<endl;
		selected_snps+=len[i];
        }
	
	cout<<" Number of selected SNPs w.r.t  annot file : " <<selected_snps<<endl;


	 step_size=Nsnp/Njack;
         step_size_rem=Nsnp%Njack;
	cout<<"Number of SNPs per block : "<<step_size<<endl;
      //  cout<<"stepsize : "<<step_size_rem<<endl;
	jack_bin.resize(Njack, vector<int>(Nbin,0));
	int temp;
	for (int i=0;i<Nsnp;i++)
	   for(int j=0;j<Nbin;j++)
		 if (annot_bool[i][j]==1){
			temp=i/step_size;
			if (temp>=Njack)
				temp=Njack-1;
			//cout<<i<<"xxx"<<j<<"xxx"<<temp<<endl;
			jack_bin[temp][j]++;	
		 }
/*	
cout<<"jackbin"<<endl;
	for (int i=0;i<Njack;i++){
	   for(int j=0;j<Nbin;j++)
                cout<<jack_bin[i][j]<<" ";
	  cout<<endl;
        }*/
/*
for (int i=0;i<linenum;i++){
  for(int j=0;j<Nbin;j++)
	cout<<annot_bool[i][j]<<" ";
  cout<<endl;
}
*/

}


//vector <int> SNP_annot;
void read_bim (string filename){
        ifstream inp(filename.c_str());
        if (!inp.is_open()){
                cerr << "Error reading file "<< filename <<endl;
                exit(1);
        }
        string line;
        int j = 0 ;
        int linenum = 0 ;
        while(std::getline (inp, line)){
                linenum ++;
                char c = line[0];
                if (c=='#')
                        continue;
                istringstream ss (line);
                if (line.empty())
                        continue;
                j++;
        }
        Nsnp = j;
        inp.close();
	cout<<"#SNP in bim file "<<Nsnp<<endl;
}





void count_pheno(std::string filename){
        ifstream ifs(filename.c_str(), ios::in);

        std::string line;
        int i=0;
        while(std::getline(ifs, line)){
                i++;
        }
        Nindv=i-1;
}




int  count_fam(std::string filename){
        ifstream ifs(filename.c_str(), ios::in);

        std::string line;
        int i=0;
        while(std::getline(ifs, line)){
                i++;
        }
        return i;
}



//// functions related to reading without mailman

template<typename T>
static std::istream & binary_read(std::istream& stream, T& value){
        return stream.read(reinterpret_cast<char*>(&value), sizeof(T));
}

void set_metadata() {
        wordsize = sizeof(char) * 8;
        unitsize = 2;
        unitsperword = wordsize/unitsize;
        mask2 = 0;
        for (int i = 0 ; i < unitsize; i++)
                mask2 = mask2 |(0x1<<i);
    nrow = Nsnp;
    ncol = ceil(1.0*Nindv/unitsperword);
}


int simulate2_geno_from_random(float p_j){
        float rval = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        float dist_pj[3] = { (1-p_j)*(1-p_j), 2*p_j*(1-p_j), p_j*p_j };
        if(rval < dist_pj[0] )
                return 0;
        else if( rval >= dist_pj[0] && rval < (dist_pj[0]+dist_pj[1]))
                return 1;
        else
                return 2;
}









float get_observed_pj(const unsigned char* line){
        int y[4];
        int observed_sum=0;
        int observed_ct=0;

        for (int k = 0 ;k < ncol ; k++) {
                unsigned char c = line [k];
                y[0] = (c)&mask2;
                y[1] = (c>>2)&mask2;
                y[2] = (c>>4)&mask2;
                y[3] = (c>>6)&mask2;
                int j0 = k * unitsperword;
                int lmax = 4;
                if (k == ncol - 1)  {
                        lmax = Nindv%4;
                        lmax = (lmax==0)?4:lmax;
                }
                for ( int l = 0 ; l < lmax; l++){
                        int j = j0 + l ;
                        // Extract  PLINK coded genotype and convert into 0/1/2
                        // // PLINK coding: 
                        // // 00->0
                        // // 01->missing
                        // // 10->1
                        // // 11->2
                        int val = y[l];
                        val-- ;
                        if(val != 0){
                                val =  (val < 0 ) ? 0 :val ;
                                observed_sum += val;
                                observed_ct ++;
                        }
                }
        }
        return observed_sum*0.5/observed_ct;

}





void read_bed2 (std::istream& ifs,bool allow_missing,int num_snp)  {
         //ifstream ifs (filename.c_str(), ios::in|ios::binary);
        char magic[3];
        set_metadata ();

    gtype =  new unsigned char[ncol];

     if(read_header)
      binary_read(ifs,magic);
 
        int sum=0;

        // Note that the coding of 0 and 2 can get flipped relative to plink because plink uses allele frequency (minor)
        // allele to code a SNP as 0 or 1.
        // This flipping does not matter for results.
        int y[4];

int bin_pointer;

vector<int> pointer_bins;

for(int i=0;i<num_snp;i++){
                global_snp_index++;
                ifs.read (reinterpret_cast<char*>(gtype), ncol*sizeof(unsigned char));
                float p_j = get_observed_pj(gtype);
	 

	   pointer_bins.clear();      
       	  for(int bin_index=0;bin_index<Nbin;bin_index++)
       		if(annot_bool[global_snp_index][bin_index]==1)
			  pointer_bins.push_back(bin_index);
			//bin_pointer=bin_index;

	  for (int k = 0 ;k < ncol ; k++) {
                unsigned char c = gtype [k];
                        // Extract PLINK genotypes
                y[0] = (c)&mask2;
                y[1] = (c>>2)&mask2;
                y[2] = (c>>4)&mask2;
                y[3] = (c>>6)&mask2;
                        int j0 = k * unitsperword;
                        // Handle number of individuals not being a multiple of 4
                        int lmax = 4;
                        if (k == ncol - 1)  {
                                lmax = Nindv%4;
                                lmax = (lmax==0)?4:lmax;
                        }
                        for ( int l = 0 ; l < lmax; l++){
                                int j = j0 + l ;
                                // Extract  PLINK coded genotype and convert into 0/1/2
                                // PLINK coding: 
                                // 00->0
                                // 01->missing
                                // 10->1
                                // 11->2
                                int val = y[l];
                                if(val==1 && !allow_missing){
                                        val = simulate2_geno_from_random(p_j);
                                        val++;
                                        val = (val==1) ? 0 : val;
                                   val=0;
                                 }
                                val-- ;
                                val =  (val < 0 ) ? 0 :val ;
                                sum += val;

                            for(int bin_index=0;bin_index<pointer_bins.size();bin_index++){
					
				       bin_pointer=pointer_bins[bin_index];
                                      int snp_index;

                                     if(use_mailman==true){
                                         snp_index=allgen_mail[bin_pointer].index;
                                         int horiz_seg_no = snp_index/allgen_mail[bin_pointer].segment_size_hori;
                                         allgen_mail[bin_pointer].p[horiz_seg_no][j] = 3 *allgen_mail[bin_pointer].p[horiz_seg_no][j]  + val;
                                     // computing sum for every snp to compute mean
                                         allgen_mail[bin_pointer].columnsum[snp_index]+=val;

                                      }
                                     else{
                                         snp_index=allgen[bin_pointer].index;
                                         allgen[bin_pointer].gen(snp_index,j)=val;
                                     }


                            }

                    }
        }

    for(int bin_index=0;bin_index<pointer_bins.size();bin_index++){
     		bin_pointer=pointer_bins[bin_index];
	         if(use_mailman==true)
                  allgen_mail[bin_pointer].index++;
               else
                   allgen[bin_pointer].index++;
    }	



}

        sum = 0 ;
        delete[] gtype;
}


int main(int argc, char const *argv[]){
 


parse_args(argc,argv);
    
        int B = command_line_opts.batchNum;
        k_orig = command_line_opts.num_of_evec ;
        debug = command_line_opts.debugmode ;
        check_accuracy = command_line_opts.getaccuracy;
        var_normalize = false;
        accelerated_em = command_line_opts.accelerated_em;
        k = k_orig + command_line_opts.l;
        k = (int)ceil(k/10.0)*10;
        command_line_opts.l = k - k_orig;
        srand((unsigned int) time(0));
	Nz=command_line_opts.num_of_evec;
        k=Nz;

	Njack=command_line_opts.jack_number;

////
string filename;
//////////////////////////// Read multi genotypes
string line;
int cov_num;
int num_files=0;
string geno_name=command_line_opts.GENOTYPE_FILE_PATH;

/////////Read bim file to count # SNPs
std::stringstream f1;
f1 << geno_name << ".bim";
read_bim (f1.str());

//////Read annotation files
filename=command_line_opts.Annot_PATH;

read_annot(filename);

///reading phnotype and save the number of indvs
std::stringstream f0;
f0 << geno_name << ".fam";
string name_fam=f0.str();
int fam_lines=count_fam(name_fam);
Nindv=fam_lines; 
//all_zb= MatrixXdr::Random(Nindv,Nz);


if(use_mailman==true) 
  allgen_mail.resize(Nbin);
 
int bin_index=0;
///// code for handeling overlapping annotations
std::stringstream f3;
f3 << geno_name << ".bed";
string name=f3.str();
cout<<name<<endl;
ifstream ifs (name.c_str(), ios::in|ios::binary);
read_header=true;
global_snp_index=-1;

if (!ifs.is_open()){
   cerr << "Error reading file "<< name  <<endl;
    exit(1);
}
	

cout<<"Start reading genotypes in blocks"<<endl;
    

for (int jack_index=0;jack_index<Njack;jack_index++){	 //// loop over number of stream blocks

	int read_Nsnp=(jack_index<(Njack-1)) ? (step_size) : (step_size+step_size_rem);
       if(use_mailman==true){
        for (int i=0;i<Nbin;i++){
	allgen_mail[i].segment_size_hori = floor(log(Nindv)/log(3)) - 2 ;
        allgen_mail[i].Nsegments_hori = ceil(jack_bin[jack_index][i]*1.0/(allgen_mail[i].segment_size_hori*1.0));
        allgen_mail[i].p.resize(allgen_mail[i].Nsegments_hori,std::vector<int>(Nindv));
        allgen_mail[i].not_O_i.resize(jack_bin[jack_index][i]);
        allgen_mail[i].not_O_j.resize(Nindv);
	allgen_mail[i].index=0;
	allgen_mail[i].Nsnp=jack_bin[jack_index][i];
	allgen_mail[i].Nindv=Nindv;
       	
	 allgen_mail[i].columnsum.resize(jack_bin[jack_index][i],1);
	  for (int index_temp=0;index_temp<jack_bin[jack_index][i];index_temp++)
		    allgen_mail[i].columnsum[index_temp]=0;

	 }
       }
       else{
	   for (int bin_index=0;bin_index<Nbin;bin_index++){
                allgen[bin_index].gen.resize(jack_bin[jack_index][bin_index],Nindv);
                allgen[bin_index].index=0;
           }
       }

       
	read_bed2(ifs,missing,read_Nsnp);   //// read a genotype block of size Nindv*read_Nsnp
       read_header=false;

     for (int bin_index=0;bin_index<Nbin;bin_index++){  /// loop over annotations defined in a stream block
	  int num_snp;
		num_snp=allgen_mail[bin_index].index;

	  if(num_snp!=0){
	  stds.resize(num_snp,1);
	  means.resize(num_snp,1);
	  
	for (int i=0;i<num_snp;i++) 						/// compute mean of SNPs
		   means(i,0)=(double)allgen_mail[bin_index].columnsum[i]/Nindv; 
          
	  for (int i=0;i<num_snp;i++)						/// compute std of SNPs
	       stds(i,0)=1/sqrt((means(i,0)*(1-(0.5*means(i,0)))));

	   	g=allgen_mail[bin_index];
                 g.segment_size_hori = floor(log(Nindv)/log(3)) - 2 ;
        	 g.Nsegments_hori = ceil(jack_bin[jack_index][bin_index]*1.0/(g.segment_size_hori*1.0));
        	 g.p.resize(g.Nsegments_hori,std::vector<int>(Nindv));
        	 g.not_O_i.resize(jack_bin[jack_index][bin_index]);
        	 g.not_O_j.resize(Nindv);
		initial_var();
		

////////////////////////////////// multiplication
	MatrixXdr temp1;
	MatrixXdr temp2;
	temp1=MatrixXdr::Random(Nindv,Nz);     /// lets asssume number of columns in temp1= number of rows in temp2=Nz, Nz is given as input argument of the code with -k
	temp2=MatrixXdr::Random(Nz,num_snp);
	MatrixXdr out1=compute_Xtz(num_snp,temp1,Nz); /// compute X^t*temp1 where X is Nindv*num_snp genotype block and temp1 is Nindv*Nz matrix
	MatrixXdr out2=compute_Xz(num_snp,temp2,Nz); /// compute X*temp2 where X is Nindv*num_snp genotype block anf temp2 is Nz*num_snp matrix
//////////////////////////////////////end
	cout<<" Reading and computing bin "<<bin_index <<"  of "<< jack_index<<"-th is finished"<<endl;
	   
	    if(use_mailman==true){
		delete[] sum_op;
        	delete[] partialsums;
       		 delete[] yint_e;
        	delete[] yint_m;
        	for (int i  = 0 ; i < hsegsize; i++)
                	delete[] y_m [i];
        	delete[] y_m;

        	for (int i  = 0 ; i < g.Nindv; i++)
                	delete[] y_e[i];
        	delete[] y_e;

        	std::vector< std::vector<int> >().swap(g.p);
        	std::vector< std::vector<int> >().swap(g.not_O_j);
        	std::vector< std::vector<int> >().swap(g.not_O_i);
		std::vector< std::vector<int> >().swap(allgen_mail[bin_index].p);
                std::vector< std::vector<int> >().swap(allgen_mail[bin_index].not_O_j);
                std::vector< std::vector<int> >().swap(allgen_mail[bin_index].not_O_i);
        //g.p.clear();
        //g.not_O_j.clear();
        //g.not_O_i.clear();
        	g.columnsum.clear();
        	g.columnsum2.clear();
        	g.columnmeans.clear();
       		 g.columnmeans2.clear();
		 allgen_mail[bin_index].columnsum.clear();
                allgen_mail[bin_index].columnsum2.clear();
                allgen_mail[bin_index].columnmeans.clear();
                 allgen_mail[bin_index].columnmeans2.clear();
	    }
        }

     }
//cout<<" Reading and computing  of "<< jack_index<<"-th is finished"<<endl;
}

cout<<" Reading and computing  of all blocks are finished"<<endl;

	return 0;
}
