#include "CvGaborFace.h"

#ifndef BUFSIZE
#define BUFSIZE 20
#endif

CvGaborFace::CvGaborFace(void)
{
	gbJet = NULL;
	imgHist = NULL;
}

CvGaborFace::~CvGaborFace(void)
{
	if(gbJet != NULL)
		delete gbJet;
	if(imgHist != NULL)
	{
		for(int i =0; i<imgNum; i++)
		{
			delete imgHist[i];
			imgHist[i] = NULL;
		}

		imgHist = NULL;

	}
}


void CvGaborFace::loadTrnList(string listName)
{
	file_lists trnImgList;

	string name;

	/***** Read in training image list ******/
	ifstream is(listName.c_str());
	string tempFile;
	string commonDirectory;

	if(!is){
		fprintf(stderr, "ERROR(%s, %d): CANNOT load model \"%s\"\n",
			__FILE__, __LINE__, "train");
		exit(0);
	}

	getline(is,commonDirectory);

	while(is.eof() != true)
	{
		getline(is,tempFile);
		getline(is,tempFile);
		trnImgList.push_back(tempFile);
		name = strtok((char *)tempFile.c_str(), "_");
		trainObjectName.push_back(name);
		getline(is,tempFile);
	}

	is.close();


	/***Init image descriptors***********/
	printf("Initializing image descriptors\n");

	gbJet = new CvGaborJet(2,4);
	gbJet->Initialize();

	imgHist = new blkHistogram * [trnImgList.size()];

	// Read in testing image

	imgNum = (int)trnImgList.size();

	for(int i =0; i<imgNum; i++)
	{
		string fullTrnName = commonDirectory+trnImgList[i];

		IplImage * orgImg= cvLoadImage(fullTrnName.c_str());

		imgHist[i] = gbJet->img2LBPHist(orgImg, 5);

		cvReleaseImage(&orgImg);

	}

	
}

int CvGaborFace::faceRecog(IplImage * input, string * name)
{
	blkHistogram * hist1 = gbJet->img2LBPHist(input, 5);

	float minDist = 100000000.0;
	float maxSim  = 0;//1000000000;
	float dist    = 0;
	int bestMatched = -1;


	float max10[BUFSIZE];
	int   index10[BUFSIZE];

	for(int i=0; i<BUFSIZE; i++)
	{
		max10[i] = 0.0;
		index10[i] = 0;
	}

	for(int i=0; i<imgNum; i++)
	{
		blkHistogram * hist2 = imgHist[i];

		dist = computeHisSimLOW(hist1, hist2);

		if(dist > maxSim)
		{
			maxSim = dist;

			bestMatched = i;

		}

		int pos = 10000;

		for(int jj=BUFSIZE-1; jj>=0; jj--)
		{
			if(dist > max10[jj])
			{
				pos = jj;
				//break;
			}
		}

		if(pos <= BUFSIZE-1)
		{
			for(int jj = BUFSIZE-1; jj>pos; jj--)
			{
				max10[jj] = max10[jj-1];
				index10[jj] = index10[jj-1];

			}

			max10[pos] = dist;
			index10[pos] = i;
		}
	}

	maxSim = 0;

	for(int ii=0; ii<BUFSIZE; ii++)
	{
		blkHistogram * hist2 = imgHist[index10[ii]];

		dist = computeHisSim(hist1, hist2);

	
		if(dist > maxSim)
		{
			maxSim = dist;

			bestMatched = index10[ii];

		}

	}

	printf("traingobject, %s, filename, %4d, maxSim is, %f\n", trainObjectName[bestMatched].c_str(),bestMatched, maxSim);

	*name = trainObjectName[bestMatched];

	if(maxSim < 4250.0)
		return 0;
	else
		return 1;


}