#include <iostream>
#include <string>

extern "C" {
#include "svm_common.h"
#include "svm_learn.h"
}

void wait_any_key()
{
	printf("\n(more)\n");
	(void)getc(stdin);
}

void print_help()
{
	printf("\nSVM-light %s: Support Vector Machine, learning module     %s\n",VERSION,VERSION_DATE);
	copyright_notice();
	printf("   usage: svm_learn [options] example_file model_file\n\n");
	printf("Arguments:\n");
	printf("         example_file-> file with training data\n");
	printf("         model_file  -> file to store learned decision rule in\n");

	printf("General options:\n");
	printf("         -?          -> this help\n");
	printf("         -v [0..3]   -> verbosity level (default 1)\n");
	printf("         -B [0,1]    -> binary input files (default 1)\n");
	printf("Learning options:\n");
	printf("         -z {c,r,p}  -> select between classification (c), regression (r),\n");
	printf("                        and preference ranking (p) (default classification)\n");
	printf("         -c float    -> C: trade-off between training error\n");
	printf("                        and margin (default [avg. x*x]^-1)\n");
	printf("         -w [0..]    -> epsilon width of tube for regression\n");
	printf("                        (default 0.1)\n");
	printf("         -j float    -> Cost: cost-factor, by which training errors on\n");
	printf("                        positive examples outweight errors on negative\n");
	printf("                        examples (default 1) (see [4])\n");
	printf("         -b [0,1]    -> use biased hyperplane (i.e. x*w+b>0) instead\n");
	printf("                        of unbiased hyperplane (i.e. x*w>0) (default 1)\n");
	printf("         -i [0,1]    -> remove inconsistent training examples\n");
	printf("                        and retrain (default 0)\n");
	printf("Performance estimation options:\n");
	printf("         -x [0,1]    -> compute leave-one-out estimates (default 0)\n");
	printf("                        (see [5])\n");
	printf("         -o ]0..2]   -> value of rho for XiAlpha-estimator and for pruning\n");
	printf("                        leave-one-out computation (default 1.0) (see [2])\n");
	printf("         -k [0..100] -> search depth for extended XiAlpha-estimator \n");
	printf("                        (default 0)\n");
	printf("Transduction options (see [3]):\n");
	printf("         -p [0..1]   -> fraction of unlabeled examples to be classified\n");
	printf("                        into the positive class (default is the ratio of\n");
	printf("                        positive and negative examples in the training data)\n");
	printf("Kernel options:\n");
	printf("         -t int      -> type of kernel function:\n");
	printf("                        0: linear (default)\n");
	printf("                        1: polynomial (s a*b+c)^d\n");
	printf("                        2: radial basis function exp(-gamma ||a-b||^2)\n");
	printf("                        3: sigmoid tanh(s a*b + c)\n");
	printf("                        4: user defined kernel from kernel.h\n");
	printf("         -d int      -> parameter d in polynomial kernel\n");
	printf("         -g float    -> parameter gamma in rbf kernel\n");
	printf("         -s float    -> parameter s in sigmoid/poly kernel\n");
	printf("         -r float    -> parameter c in sigmoid/poly kernel\n");
	printf("         -u string   -> parameter of user defined kernel\n");
	printf("Optimization options (see [1]):\n");
	printf("         -q [2..]    -> maximum size of QP-subproblems (default 10)\n");
	printf("         -n [2..q]   -> number of new variables entering the working set\n");
	printf("                        in each iteration (default n = q). Set n<q to prevent\n");
	printf("                        zig-zagging.\n");
	printf("         -m [5..]    -> size of cache for kernel evaluations in MB (default 40)\n");
	printf("                        The larger the faster...\n");
	printf("         -e float    -> eps: Allow that error for termination criterion\n");
	printf("                        [y [w*x+b] - 1] >= eps (default 0.001)\n");
	printf("         -y [0,1]    -> restart the optimization from alpha values in file\n");
	printf("                        specified by -a option. (default 0)\n");
	printf("         -h [5..]    -> number of iterations a variable needs to be\n"); 
	printf("                        optimal before considered for shrinking (default 100)\n");
	printf("         -f [0,1]    -> do final optimality check for variables removed\n");
	printf("                        by shrinking. Although this test is usually \n");
	printf("                        positive, there is no guarantee that the optimum\n");
	printf("                        was found if the test is omitted. (default 1)\n");
	printf("         -y string   -> if option is given, reads alphas from file with given\n");
	printf("                        and uses them as starting point. (default 'disabled')\n");
	printf("         -# int      -> terminate optimization, if no progress after this\n");
	printf("                        number of iterations. (default 100000)\n");
	printf("Output options:\n");
	printf("         -l string   -> file to write predicted labels of unlabeled\n");
	printf("                        examples into after transductive learning\n");
	printf("         -a string   -> write all alphas to this file after learning\n");
	printf("                        (in the same order as in the training set)\n");
	wait_any_key();
	printf("\nMore details in:\n");
	printf("[1] T. Joachims, Making Large-Scale SVM Learning Practical. Advances in\n");
	printf("    Kernel Methods - Support Vector Learning, B. Schölkopf and C. Burges and\n");
	printf("    A. Smola (ed.), MIT Press, 1999.\n");
	printf("[2] T. Joachims, Estimating the Generalization performance of an SVM\n");
	printf("    Efficiently. International Conference on Machine Learning (ICML), 2000.\n");
	printf("[3] T. Joachims, Transductive Inference for Text Classification using Support\n");
	printf("    Vector Machines. International Conference on Machine Learning (ICML),\n");
	printf("    1999.\n");
	printf("[4] K. Morik, P. Brockhausen, and T. Joachims, Combining statistical learning\n");
	printf("    with a knowledge-based approach - A case study in intensive care  \n");
	printf("    monitoring. International Conference on Machine Learning (ICML), 1999.\n");
	printf("[5] T. Joachims, Learning to Classify Text Using Support Vector\n");
	printf("    Machines: Methods, Theory, and Algorithms. Dissertation, Kluwer,\n");
	printf("    2002.\n\n");
}

void read_input_parameters(LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm)
{
	long i;
	char type[100];

	/* set default */
	strcpy (learn_parm->predfile, "trans_predictions");
	strcpy (learn_parm->alphafile, "");
	learn_parm->biased_hyperplane=1;
	learn_parm->sharedslack=0;
	learn_parm->remove_inconsistent=0;
	learn_parm->skip_final_opt_check=0;
	learn_parm->svm_maxqpsize=10;
	learn_parm->svm_newvarsinqp=0;
	learn_parm->svm_iter_to_shrink=-9999;
	learn_parm->maxiter=10000;
	learn_parm->kernel_cache_size=40;
	//learn_parm->svm_c=0.0;
	learn_parm->svm_c=1000;

	learn_parm->eps=0.1;
	learn_parm->transduction_posratio=-1.0;
	learn_parm->svm_costratio=1.0;
	learn_parm->svm_costratio_unlab=1.0;
	learn_parm->svm_unlabbound=1E-5;
	learn_parm->epsilon_crit=0.001;
	learn_parm->epsilon_a=1E-15;
	learn_parm->compute_loo=0;
	learn_parm->rho=1.0;
	learn_parm->xa_depth=0;
	kernel_parm->kernel_type=0;//0:linear,2 rbf kernel type
	kernel_parm->poly_degree=3;
	kernel_parm->rbf_gamma=1.0;
	kernel_parm->coef_lin=1;
	kernel_parm->coef_const=1;
	strcpy(kernel_parm->custom,"empty");
	strcpy(type,"c");

	if(learn_parm->svm_iter_to_shrink == -9999) {
		if(kernel_parm->kernel_type == LINEAR) 
			learn_parm->svm_iter_to_shrink=2;
		else
			learn_parm->svm_iter_to_shrink=100;
	}
	if(strcmp(type,"c")==0) {
		learn_parm->type=CLASSIFICATION;
	}
	else if(strcmp(type,"r")==0) {
		learn_parm->type=REGRESSION;
	}
	else if(strcmp(type,"p")==0) {
		learn_parm->type=RANKING;
	}
	else if(strcmp(type,"o")==0) {
		learn_parm->type=OPTIMIZATION;
	}
	else if(strcmp(type,"s")==0) {
		learn_parm->type=OPTIMIZATION;
		learn_parm->sharedslack=1;
	}
	else {
		printf("\nUnknown type '%s': Valid types are 'c' (classification), 'r' regession, and 'p' preference ranking.\n",type);
		wait_any_key();
		print_help();
		exit(0);
	}    
	if((learn_parm->skip_final_opt_check) 
		&& (kernel_parm->kernel_type == LINEAR)) {
			printf("\nIt does not make sense to skip the final optimality check for linear kernels.\n\n");
			learn_parm->skip_final_opt_check=0;
	}    
	if((learn_parm->skip_final_opt_check) 
		&& (learn_parm->remove_inconsistent)) {
			printf("\nIt is necessary to do the final optimality check when removing inconsistent \nexamples.\n");
			wait_any_key();
			print_help();
			exit(0);
	}    
	if((learn_parm->svm_maxqpsize<2)) {
		printf("\nMaximum size of QP-subproblems not in valid range: %ld [2..]\n",learn_parm->svm_maxqpsize); 
		wait_any_key();
		print_help();
		exit(0);
	}
	if((learn_parm->svm_maxqpsize<learn_parm->svm_newvarsinqp)) {
		printf("\nMaximum size of QP-subproblems [%ld] must be larger than the number of\n",learn_parm->svm_maxqpsize); 
		printf("new variables [%ld] entering the working set in each iteration.\n",learn_parm->svm_newvarsinqp); 
		wait_any_key();
		print_help();
		exit(0);
	}
	if(learn_parm->svm_iter_to_shrink<1) {
		printf("\nMaximum number of iterations for shrinking not in valid range: %ld [1,..]\n",learn_parm->svm_iter_to_shrink);
		wait_any_key();
		print_help();
		exit(0);
	}
	if(learn_parm->svm_c<0) {
		printf("\nThe C parameter must be greater than zero!\n\n");
		wait_any_key();
		print_help();
		exit(0);
	}
	if(learn_parm->transduction_posratio>1) {
		printf("\nThe fraction of unlabeled examples to classify as positives must\n");
		printf("be less than 1.0 !!!\n\n");
		wait_any_key();
		print_help();
		exit(0);
	}
	if(learn_parm->svm_costratio<=0) {
		printf("\nThe COSTRATIO parameter must be greater than zero!\n\n");
		wait_any_key();
		print_help();
		exit(0);
	}
	if(learn_parm->epsilon_crit<=0) {
		printf("\nThe epsilon parameter must be greater than zero!\n\n");
		wait_any_key();
		print_help();
		exit(0);
	}
	if(learn_parm->rho<0) {
		printf("\nThe parameter rho for xi/alpha-estimates and leave-one-out pruning must\n");
		printf("be greater than zero (typically 1.0 or 2.0, see T. Joachims, Estimating the\n");
		printf("Generalization Performance of an SVM Efficiently, ICML, 2000.)!\n\n");
		wait_any_key();
		print_help();
		exit(0);
	}
	if((learn_parm->xa_depth<0) || (learn_parm->xa_depth>100)) {
		printf("\nThe parameter depth for ext. xi/alpha-estimates must be in [0..100] (zero\n");
		printf("for switching to the conventional xa/estimates described in T. Joachims,\n");
		printf("Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)\n");
		wait_any_key();
		print_help();
		exit(0);
	}
}

void trainmodel(char*docfile,char* modelfile )
{

	//char* docfile ;
	//char* modelfile ; 
	long verbosity = 1;
	long format    = 1;

	/*std::cout<<sys.model_file<<std::endl;
	int len1 = sys.train_file.length();
	int len2 = sys.model_file.length();
	int len3 = sys.predit_file.length();

	char* docfile ;
	char* modelfile ; 
	docfile   = new char[len1+1];
	modelfile = new char[len2+1];
	sys.train_file.copy(docfile,len1,0);
	sys.model_file.copy(modelfile,len2,0);
	docfile[len1] = 0;
	modelfile[len2] = 0;

	long verbosity = sys.verbosity;
	long format    = sys.format;*/



	MODEL *model = (MODEL*)my_malloc(sizeof(MODEL));
	DOC **docs;  /* training examples */
	long totwords,totdoc,i;
	double *target;
	double *alpha_in=NULL;
	KERNEL_CACHE *kernel_cache;
	LEARN_PARM learn_parm;
	KERNEL_PARM kernel_parm;

	char restartfile[10] = "";       /* file with initial alphas */

	/* Support for binary input file added by N. Dalal*/
	read_input_parameters(&learn_parm,&kernel_parm);

	/* Support for binary input file added by N. Dalal*/
	if (format) {
		read_binary_documents(docfile,&docs,&target,&totwords,&totdoc);
	} else {
		read_documents(docfile,&docs,&target,&totwords,&totdoc);
	}

	if(restartfile[0]) alpha_in=read_alphas(restartfile,totdoc);

	if(kernel_parm.kernel_type == LINEAR) { /* don't need the cache */
		kernel_cache=NULL;
	}
	else {
		/* Always get a new kernel cache. It is not possible to use the
		same cache for two different training runs */
		kernel_cache=kernel_cache_init(totdoc,learn_parm.kernel_cache_size);
	}

	if(learn_parm.type == CLASSIFICATION) {
		svm_learn_classification(docs,target,totdoc,totwords,&learn_parm,
			&kernel_parm,kernel_cache,model,alpha_in);
	}
	else if(learn_parm.type == REGRESSION) {
		svm_learn_regression(docs,target,totdoc,totwords,&learn_parm,
			&kernel_parm,&kernel_cache,model);
	}
	else if(learn_parm.type == RANKING) {
		svm_learn_ranking(docs,target,totdoc,totwords,&learn_parm,
			&kernel_parm,&kernel_cache,model);
	}
	else if(learn_parm.type == OPTIMIZATION) {
		svm_learn_optimization(docs,target,totdoc,totwords,&learn_parm,
			&kernel_parm,kernel_cache,model,alpha_in);
	}

	if(kernel_cache) {
		/* Free the memory used for the cache. */
		kernel_cache_cleanup(kernel_cache);
	}

	/* Warning: The model contains references to the original data 'docs'.
	If you want to free the original data, and only keep the model, you 
	have to make a deep copy of 'model'. */
	/* deep_copy_of_model=copy_model(model); */
	if (format) {
		write_binary_model(modelfile,model);
	} else {
		write_model(modelfile,model);
	}

	free(alpha_in);
	free_model(model,0);
	for(i=0;i<totdoc;i++) 
		free_example(docs[i],1);
	free(docs);
	free(target);

	/*delete []docfile;
	delete []modelfile;*/
}
	


