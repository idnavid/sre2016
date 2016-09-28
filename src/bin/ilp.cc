#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <math.h>
#include "ilp.h"


namespace kaldi{

GlpkILP::GlpkILP(BaseFloat delta) {
	this->_delta = delta;
}


GlpkILP::GlpkILP(Matrix<BaseFloat>& distanceMatrix, BaseFloat delta) {
	this->_distanceMatrix = distanceMatrix;
	this->_delta = delta;
}


void GlpkILP::glpkIlpProblem() {
	this->_problem.push_back("Minimize");
	this->_problem.push_back(problemMinimize());
	this->_problem.push_back("Subject To");
	problemConstraintsColumnSum();
	problemConstraintsCenter();
	//distanceUpperBound();
	this->_problem.push_back("Binary");
	listBinaryVariables();
	this->_problem.push_back("End");
}

void GlpkILP::distanceUpperBound() {
	int32 n = this->_distanceMatrix.NumRows();
	for (size_t i = 0; i < this->_distanceMatrix.NumRows(); i++) {
		for (size_t j = 0; j < this->_distanceMatrix.NumRows(); j++) {
			if (i != j) {
				BaseFloat d = this->_distanceMatrix(i, j);
				std::string constraint = "C" + \
						numberToString(n) + \
				 		": " + numberToString(d) + " " + indexToVarName("x",i,j);
				constraint += " < " + numberToString(this->_delta);
				this->_problem.push_back(constraint);
				n++;
			}
		}
	}
}

std::string GlpkILP::problemMinimize() {
	std::string objective = "problem : " + indexToVarName("x",0,0);
	for (size_t i = 1; i < this->_distanceMatrix.NumRows(); i++) {
		objective += " + " + indexToVarName("x",i,i);
	}
	for (size_t i = 0; i < this->_distanceMatrix.NumRows(); i++) {
		for (size_t j = 0; j < this->_distanceMatrix.NumRows(); j++) {
			if (i != j) {
				BaseFloat d = this->_distanceMatrix(i, j) / this->_delta;
				// Why did we need to assure d > 0 ?
				if ((d > 0) && (d <= 1)) {
					objective += " + " + numberToString(d) + " " + indexToVarName("x",i,j);
				}
			} 
		}
	}
	return objective;
}


void GlpkILP::problemConstraintsColumnSum() {
	for (size_t i = 0; i < this->_distanceMatrix.NumRows(); i++) {
		std::string constraint = "C" + numberToString(i) + ": " + indexToVarName("x",i,i);
		for (size_t j = 0; j < this->_distanceMatrix.NumRows(); j++) {
			if (i != j) {
				BaseFloat d = this->_distanceMatrix(i, j);
				if (d <= this->_delta) {
					constraint += " + " + indexToVarName("x",i,j);
				}
			}
		}
		constraint += " = 1 ";
		this->_problem.push_back(constraint);
	}
}


void GlpkILP::problemConstraintsCenter() {
	for (size_t i = 0; i < this->_distanceMatrix.NumRows(); i++) {
		for (size_t j = 0; j < this->_distanceMatrix.NumRows(); j++) {
			if (i != j) {
				BaseFloat d = this->_distanceMatrix(i, j);
				if (d <= this->_delta) {
					this->_problem.push_back(indexToVarName("x",i,j) + " - " + indexToVarName("x",j,j) + " <= 0");
				}
			}
		}
	}
}


void GlpkILP::listBinaryVariables() {
	for (size_t i = 0; i < this->_distanceMatrix.NumRows(); i++) {
		for (size_t j = 0; j < this->_distanceMatrix.NumRows(); j++) {
			BaseFloat d = this->_distanceMatrix(i,j);
			if (d <= this->_delta) {
				this->_problem.push_back(indexToVarName("x",i,j));
			}
		}
	}
}

std::string GlpkILP::indexToVarName(std::string prefix, int32 i, int32 j) { 
    return prefix + "_" + numberToString(i) + "_" + numberToString(j);
}


std::vector<int32> GlpkILP::varNameToIndex(std::string& variableName){
    std::vector<std::string> fields = split(variableName, '_');
    std::vector<int32> indexes;
    indexes.push_back(std::atoi(fields[1].c_str()));
    indexes.push_back(std::atoi(fields[2].c_str()));
    return indexes;
}


void GlpkILP::Write(std::string outName){
	std::ofstream fout;
	fout.open(outName.c_str());
	for (size_t i =0; i < this->_problem.size(); i++){
		fout << this->_problem[i] << "\n";
	}
	fout.close();
}


std::vector<std::string> GlpkILP::ReadGlpkSolution(std::string glpkSolutionFile) {
	std::ifstream fin;
    fin.open(glpkSolutionFile.c_str());
    std::string line;
    std::vector<std::string> IlpClusterLabel;
    while (std::getline(fin, line)){
        if (line.find("*") != std::string::npos){
            std::vector<std::string> fields = split(line, ' ');
            std::vector<string> nonEmptyFields = returnNonEmptyFields(fields);
            std::vector<int32> varIndex = varNameToIndex(nonEmptyFields[1]);
            int32 k = varIndex[0];
            int32 j = varIndex[1];
            if (k==j) {
            	if (nonEmptyFields[3] == "1") {
                	IlpClusterLabel.push_back(numberToString(k));
                } else {
                	IlpClusterLabel.push_back(numberToString(-1));
                }
            }

            if (k!=j && nonEmptyFields[3] == "1") {
                 IlpClusterLabel[k] = numberToString(j);
            }
        }
    }

    return IlpClusterLabel;
}


}