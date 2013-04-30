#include <iostream>
#include <string>

extern "C" {
#include "svm_common.h"
}

void typeid_verbose(int mtypeid) ;
void test(char *docfile,char*modelfile)
{
	//int len1 = sys.train_file.length();
	//int len2 = sys.model_file.length();
	//int len3 = sys.predit_file.length();


	//char* docfile ;//= sys.train_file; 
	//char* modelfile ;//= sys.model_file; 
	//char* predictionsfile;// = sys.predit_file;


	//docfile   = new char[len1+1];
	//modelfile = new char[len2+1];
	//predictionsfile = new char[len3+1];

	//sys.train_file.copy(docfile,len1,0);
	//sys.model_file.copy(modelfile,len2,0);
	//sys.predit_file.copy(predictionsfile,len3,0);

	//docfile[len1] = 0;
	//modelfile[len2] = 0;
	//predictionsfile[len3] = 0;
	char * predictionsfile = "../predict.txt";

	verbosity = 1;
	format = 1;
	long pred_format = 1;

	MODEL *model;

	DOC *doc;   /* test example */
	Word *words;
	long max_docs,max_words_doc,lld;
	long totdoc=0,queryid,slackid;
	long correct=0,incorrect=0,no_accuracy=0;
	long res_a=0,res_b=0,res_c=0,res_d=0,wnum;
	long j;
	double t1,runtime=0;
	double dist,doc_label,costfactor;
	char *line,*comment; 
	FILE *predfl,*docfl;

	if (format) {
		model=read_binary_model(modelfile);
	} else {
		model=read_model(modelfile);
		if(model->kernel_parm.kernel_type == 0) { /* linear kernel */
			/* compute weight vector */
			add_weight_vector_to_linear_model(model);
		}
	}

	if(verbosity>=2) {
		printf("Classifying test examples.."); fflush(stdout);
	}

	if ((predfl = fopen (predictionsfile, "w")) == NULL)
	{ perror (predictionsfile); exit (1); }

	/* Support for binary input file added by N. Dalal*/
	if (format) {
		if ((docfl = fopen (docfile, "rb")) == NULL)
		{ perror (docfile); exit (1); }

		/* read version number */
		int version = 0;
		if (!fread (&version,sizeof(int),1,docfl))
		{ perror ("Unable to read version number"); exit (1); }

		int data_typeid = 0, target_typeid = 0;
		if (!fread (&data_typeid,sizeof(int),1,docfl))
		{ perror ("Unable to read data type id"); exit (1); }
		if (!fread (&target_typeid,sizeof(int),1,docfl))
		{ perror ("Unable to read target type id"); exit (1); }

		if(verbosity>=1) {
			typeid_verbose(data_typeid);
			typeid_verbose(target_typeid);
		}

		/* scan size of input file */
		int feature_length = 0, num_feature = 0;
		if (!fread (&num_feature,sizeof(int),1,docfl))
		{ perror ("Unable to read number of feature"); exit (1); }

		if (!fread (&feature_length,sizeof(int),1,docfl))
		{ perror ("Unable to read feature vector length"); exit (1); }

		max_words_doc = feature_length; max_docs = num_feature;
		if(verbosity>=1) {
			printf("Feature length %d, Feature count %d\n",
				feature_length,num_feature);
		}

		if(max_words_doc > MAXFEATNUM) {
			printf("\nMaximum feature number exceeds limit defined in MAXFEATNUM!\n");
			exit(1);
		}
		/* set comment to something for time being */
		comment = (char*) my_malloc(sizeof(char)*1);
		*comment = 0;

		words = (Word *)my_malloc(sizeof(Word)*(max_words_doc+1));
		totdoc=0;

		if(verbosity>=2) {
			printf("Reading examples into memory..."); fflush(stdout);
		}

		while(!feof(docfl) && totdoc < max_docs) {
			/* wnum contains type id for time being*/
			if(!read_feature(docfl,words,&doc_label,
				target_typeid, data_typeid,
				&queryid,&slackid,&costfactor,
				&wnum,max_words_doc,&comment)) 
			{
				printf("\nParsing error in vector %ld!\n",totdoc);
				exit(1);
			}
			totdoc++;  
			if(model->kernel_parm.kernel_type == 0) {   /* linear kernel */
				doc = create_example(-1,0,0,0.0,
					create_svector(words,feature_length,comment,1.0));
				t1=get_runtime();
				dist=classify_example_linear(model,doc);
				runtime+=(get_runtime()-t1);
				free_example(doc,1);
			}
			else {                             /* non-linear kernel */
				doc = create_example(-1,0,0,0.0,
					create_svector(words,feature_length,comment,1.0));
				t1=get_runtime();
				dist=classify_example(model,doc);
				runtime+=(get_runtime()-t1);
				free_example(doc,1);
			}
			if(dist>0) {
				if(pred_format==0) { /* old weired output format */
					fprintf(predfl,"%.8g:+1 %.8g:-1\n",dist,-dist);
				}
				if(doc_label>0) correct++; else incorrect++;
				if(doc_label>0) res_a++; else res_b++;
			}
			else {
				if(pred_format==0) { /* old weired output format */
					fprintf(predfl,"%.8g:-1 %.8g:+1\n",-dist,dist);
				}
				if(doc_label<0) correct++; else incorrect++;
				if(doc_label>0) res_c++; else res_d++;
			}
			if(pred_format==1) { /* output the value of decision function */
				fprintf(predfl,"%.8g\n",dist);
			}
			if((int)(0.01+(doc_label*doc_label)) != 1) 
			{ no_accuracy=1; } /* test data is not binary labeled */
			if(verbosity>=2) {
				if(totdoc % 100 == 0) {
					printf("%ld..",totdoc); fflush(stdout);
				}
			}
		}  
		fclose(docfl);
	} else {
		nol_ll(docfile,&max_docs,&max_words_doc,&lld); /* scan size of input file */
		max_words_doc+=2;
		lld+=2;

		line = (char *)my_malloc(sizeof(char)*lld);
		words = (Word *)my_malloc(sizeof(Word)*(max_words_doc+10));

		if ((docfl = fopen (docfile, "r")) == NULL)
		{ perror (docfile); exit (1); }
		if ((predfl = fopen (predictionsfile, "w")) == NULL)
		{ perror (predictionsfile); exit (1); }

		while((!feof(docfl)) && fgets(line,(int)lld,docfl)) {
			if(line[0] == '#') continue;  /* line contains comments */
			parse_document(line,words,&doc_label,&queryid,&slackid,&costfactor,&wnum,
				max_words_doc,&comment);
			totdoc++;
			if(model->kernel_parm.kernel_type == 0) {   /* linear kernel */
				for(j=0;(words[j]).wnum != 0;j++) {  /* Check if feature numbers   */
					if((words[j]).wnum>model->totwords) /* are not larger than in     */
						(words[j]).wnum=0;               /* model. Remove feature if   */
				}                                        /* necessary.                 */
				doc = create_example(-1,0,0,0.0,
					create_svector(words,max_words_doc,comment,1.0));
				t1=get_runtime();
				dist=classify_example_linear(model,doc);
				runtime+=(get_runtime()-t1);
				free_example(doc,1);
			}
			else {                             /* non-linear kernel */
				doc = create_example(-1,0,0,0.0,
					create_svector(words,max_words_doc,comment,1.0));
				t1=get_runtime();
				dist=classify_example(model,doc);
				runtime+=(get_runtime()-t1);
				free_example(doc,1);
			}
			if(dist>0) {
				if(pred_format==0) { /* old weired output format */
					fprintf(predfl,"%.8g:+1 %.8g:-1\n",dist,-dist);
				}
				if(doc_label>0) correct++; else incorrect++;
				if(doc_label>0) res_a++; else res_b++;
			}
			else {
				if(pred_format==0) { /* old weired output format */
					fprintf(predfl,"%.8g:-1 %.8g:+1\n",-dist,dist);
				}
				if(doc_label<0) correct++; else incorrect++;
				if(doc_label>0) res_c++; else res_d++;
			}
			if(pred_format==1) { /* output the value of decision function */
				fprintf(predfl,"%.8g\n",dist);
			}
			if((int)(0.01+(doc_label*doc_label)) != 1) 
			{ no_accuracy=1; } /* test data is not binary labeled */
			if(verbosity>=2) {
				if(totdoc % 100 == 0) {
					printf("%ld..",totdoc); fflush(stdout);
				}
			}
		}  
		free(line);
	}
	free(words);
	free_model(model,1);

	if(verbosity>=2) {
		printf("done\n");

		/*   Note by Gary Boone                     Date: 29 April 2000        */
		/*      o Timing is inaccurate. The timer has 0.01 second resolution.  */
		/*        Because classification of a single vector takes less than    */
		/*        0.01 secs, the timer was underflowing.                       */
		printf("Runtime (without IO) in cpu-seconds: %.2f\n",
			(float)(runtime/100.0));

	}
	if((!no_accuracy) && (verbosity>=1)) {
		printf("Accuracy on test set: %.2f%% (%ld correct, %ld incorrect, %ld total)\n",(float)(correct)*100.0/totdoc,correct,incorrect,totdoc);
		printf("Precision/recall on test set: %.2f%%/%.2f%%\n",(float)(res_a)*100.0/(res_a+res_b),(float)(res_a)*100.0/(res_a+res_c));
	}
	fclose(predfl);//add by rxb!!

	/*delete []docfile;
	delete []modelfile;
	delete []predictionsfile;*/
	//system("pause");
}

