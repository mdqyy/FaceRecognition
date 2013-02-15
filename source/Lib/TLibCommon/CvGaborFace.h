#pragma once

#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include "CvGaborJet.h"
#include "global.h"
#include <fstream>

using namespace std;

/** \class CvGaborFace
 *  \brief CvGaborFace class
 *  see member functions for detail
 */
class CvGaborFace
{

public:
	CvGaborFace(void);
	~CvGaborFace(void);

    /** \brief Load known face images to memory
     * Load known face images to memory.
     * @param listName A text file name where the images' name are contained.
     */
	void loadTrnList(string listName);

    /** \brief Identify the input image
     * Identify the input image by comparing the input image to the known face
     * images in the memory, then output the name of known face.
     * @param input an IplImage type input image to be identified.
     * @param name a string type containing the name of the input face image.
     * @return The test results
     */
	int faceRecog(IplImage * input, string * name);

private:
	file_lists trainObjectName;
	CvGaborJet * gbJet ;
	blkHistogram ** imgHist;	

	int imgNum;
};
