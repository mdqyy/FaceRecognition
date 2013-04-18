#ifndef _GET_CONFIG_H_ 
#define _GET_CONFIG_H_ 
  
#include <string> 
#include <map> 
using namespace std; 
  
#define COMMENT_CHAR '#' 
  
bool ReadConfig(const string & filename, map<string, string> & m); 
void PrintConfig(const map<string, string> & m); 
bool LoadParm( const map<string, string> & m, int &s, string str);
bool LoadParmBool( const map<string, string> & m, bool &s, string str);
#endif