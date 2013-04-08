

#include <cv.h>
#include <time.h>
#include <highgui.h>
#include <math.h>
//#include "stdafx.h"
#include <windows.h>
#include <mmsystem.h>
#include <stdio.h>

#include "Global.h"
#include "Define.h"

#include "features.h"


void rgb_to_hsv(int r, int g, int b, float *h, float *s, float *v)
{
   float min, max, delta, rc, gc, bc;

   rc = (float)r / 255.0;
   gc = (float)g / 255.0;
   bc = (float)b / 255.0;

   max = rc; 
   if(gc > max) max = gc;
   if(bc > max) max = bc;

   min = rc;
   if(gc < min) min = gc;
   if(bc < min) min = bc;

   delta = max - min;
   *v = max;

   if (max != 0.0)
      *s = delta / max;
   else
      *s = 0.0;

   if (*s == 0.0) {
      *h = 0.0; 
   }
   else {
      if (rc == max)
	 *h = (gc - bc) / delta;
      else if (gc == max)
	 *h = 2 + (bc - rc) / delta;
      else if (bc == max)
	 *h = 4 + (rc - gc) / delta;

      *h *= 60.0;
      if (*h < 0)
	 *h += 360.0;
    }
}



void extractImageFeatures_Type1(unsigned char * image, int R0, int C0, int R1, int C1, int widthStep, 
						  double * feature, int featureSize)
{
	int r, c, i, idx;
	int cR, cG, cB;
	float h, s, v;
	unsigned char *ptr;
	int ct;

	for(i=0; i<featureSize; i++)	feature[i] = 0;

	ct = 0;
	for(r=R0; r<R1; r++)
	for(c=C0; c<C1; c++)
	{
		ptr = image + r * widthStep + c*3;
		cB = ptr[0];
		cG = ptr[1];
		cR = ptr[2];

		if((cR == 0) && (cB==0) && (cG==0))
		{
			ct = ct;
		}
		else
		{

			rgb_to_hsv(cR, cG, cB, &h, &s, &v);

			idx = cR / 16;
			feature[idx] = feature[idx] + 1;

			idx = cG / 16 + 16;
			feature[idx] = feature[idx] + 1;

			idx = cB / 16 + 32;
			feature[idx] = feature[idx] + 1;

			h = h / 360;

			idx = (int)(h * 16); 
			if(idx > 15) idx = 15;
			idx = idx + 48;
			feature[idx] = feature[idx] + 1;

			idx = (int)(s * 16); 
			if(idx > 15) idx = 15;
			idx = idx + 64;
			feature[idx] = feature[idx] + 1;

			idx = (int)(v * 16); 
			if(idx > 15) idx = 15;
			idx = idx + 80;
			feature[idx] = feature[idx] + 1;

			//additional features
			idx = ((cR - cG) + 255) / 32;
			if(idx > 15) idx = 15;
			idx = idx + 96;
			feature[idx] = feature[idx] + 1;

			idx = ((cR - cB) + 255) / 32;
			if(idx > 15) idx = 15;
			idx = idx + 112;
			feature[idx] = feature[idx] + 1;

			idx = ((cG - cB) + 255) / 32;
			if(idx > 15) idx = 15;
			idx = idx + 128;
			feature[idx] = feature[idx] + 1;


			ct++;
		}

	}

	//normalize
	for(i=0; i<featureSize; i++)
		feature[i] = 100 * feature[i] / ct;

	return;
}