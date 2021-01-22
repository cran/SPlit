// [[Rcpp::plugins("cpp11")]]
#include <vector>
#include <Rcpp.h>
#include <algorithm>
#include "nanoflann.hpp"


class DF
{
private:
	Rcpp::NumericMatrix df_;

public:
	void importData(Rcpp::NumericMatrix& df)
	{
		df_ = df;
	}

	inline std::size_t kdtree_get_point_count() const
	{
		return df_.rows();
	}

	inline double kdtree_get_pt(const std::size_t idx, const std::size_t dim) const 
	{
		return df_(idx, dim);
	}

	template <class BBOX>
	bool kdtree_get_bbox(BBOX&) const 
	{ 
		return false; 
	}
};


typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<nanoflann::L2_Adaptor<double, DF>, DF, -1, std::size_t> kdTree;


class KDTree
{
private:
	const std::size_t dim_;
	DF data_;
	Rcpp::NumericMatrix sp_;

public:
	KDTree(Rcpp::NumericMatrix& data, Rcpp::NumericMatrix& sp) : dim_(data.cols())
	{
		if(static_cast<unsigned int>(sp.cols()) != dim_)
			Rcpp::Rcerr << "\nDimensions do not match.\n";
		else
		{
			data_.importData(data);
			sp_ = sp;
		}
	}

	std::vector<std::size_t> subsample_indices_sequential()
	{
		kdTree tree(dim_, data_, nanoflann::KDTreeSingleIndexAdaptorParams(32));
		nanoflann::KNNResultSet<double> resultSet(1);
		std::size_t index;
		double distance;
		std::vector<std::size_t> indices;
		std::size_t n = sp_.rows();
		indices.reserve(n);
		for(std::size_t i = 0; i < n; i++)
		{
			resultSet.init(&index, &distance);
			Rcpp::NumericVector row = sp_.row(i);
			std::vector<double> query_point(row.begin(), row.end());
			tree.findNeighbors(resultSet, &query_point[0], nanoflann::SearchParams());
			indices.push_back(index + 1);
			tree.removePoint(index);
		}

		return indices;
	}
};


// [[Rcpp::export]]
std::vector<std::size_t> subsample(Rcpp::NumericMatrix& data, Rcpp::NumericMatrix& points)
{
	KDTree kdt(data, points);
	return kdt.subsample_indices_sequential();
}











































