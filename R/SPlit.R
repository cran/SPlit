sp = function(n, p, dist.samp, num.subsamp=min(10000, nrow(dist.samp)), rnd.flg=ifelse(num.subsamp <= 10000, FALSE, TRUE), 
	iter.max=500, wts=NA, tol=1e-10, n0=n*p, nThreads)
{
	if(!rnd.flg)
	{
		num.subsamp = nrow(dist.samp)
	}

	if(anyNA(wts))
	{
		wts = rep(1.0, nrow(dist.samp))
    }
    else
    {
    	wts = nrow(dist.samp) * wts;
    }

	if(missing(nThreads))
	{
		num.cores = parallel::detectCores()
	}
	else
	{
		if(nThreads < 1)
		{
			stop("nThreads should be at least 1.")
		}

		threads = parallel::detectCores()
		if(nThreads > threads)
		{
			num.cores = threads
		}
		else
		{
			num.cores = nThreads
		}
	}

	bd = matrix(NA, nrow=p, ncol=2, byrow=T)
	for(i in 1:p)
	{
        bd[i, ] = range(dist.samp[, i])
    }

    if (anyDuplicated(dist.samp))
    {
	      dist.samp = jitter(dist.samp)
	      for(i in 1:p)
	      { 
	        dist.samp[, i] = pmin(pmax(dist.samp[, i], bd[i, 1]), bd[i, 2])
	      }
    }
    
	ini = matrix(jitter(dist.samp[sample(1:nrow(dist.samp), n, F), ]), ncol=p)
	for(i in 1:p)
	{ 
		ini[, i] = pmin(pmax(ini[, i], bd[i, 1]), bd[i, 2])
	}	

    if(p == 1)
    {
		ini = matrix(ini, ncol=1)
    }

    sp_ = sp_cpp(n, p, ini, dist.samp, TRUE, bd, num.subsamp, iter.max, tol, num.cores, n0, wts, rnd.flg)
    return(sp_);
}


#' Split a dataset for training and testing
#' 
#' \code{SPlit()} implements the optimal data splitting procedure described in Joseph and Vakayil (2020). 'SPlit' can be applied to both regression and classification problems, and is model-independent. As a preprocessing step, the nominal categorical columns in the dataset must be declared as factors, and the ordinal categorical columns must be converted to numeric using scoring.
#'
#' @param data The dataset including both the predictors and response(s); should not contain missing values, and only numeric and/or factor column(s) are allowed.  
#' @param splitRatio The ratio in which the dataset is to be split; should be in (0, 1) e.g. for an 80-20 split, the \code{splitRatio} is either 0.8 or 0.2.
#' @param maxIterations The maximum number of iterations before the tolerance level is reached during support points optimization.
#' @param tolerance The tolerance level for support points optimization; measured in terms of the maximum point-wise difference in distance between successive solutions.
#' @param nThreads Number of threads to be used for parallel computation; if not supplied, \code{nThreads} defaults to maximum available threads.
#'
#' @return Indices of the smaller subset in the split.
#'
#' @details Support points are defined only for continuous variables. The categorical variables are handled as follows. \code{SPlit()} will automatically convert a nominal categorical variable with \eqn{m} levels to \eqn{m-1} continuous variables using Helmert coding. Ordinal categorical variables should be converted to numerical columns using a scoring method before using \code{SPlit()}. 
#' For example, if the three levels of an ordinal variable are poor, good, and excellent, then the user may choose 1, 2, and 5 to represent the three levels. These values depend on the problem and data collection method, and therefore, \code{SPlit()} will not do it automatically. The columns of the resulting numeric dataset are then standardized to have mean zero and variance one. 
#' \code{SPlit()} then computes the support points and calls the provided \code{subsample()} function to perform a nearest neighbor subsampling. The indices of this subsample are returned.
#'
#' @export
#' @examples
#' ## 1. An 80-20 split of a numeric dataset
#' X = rnorm(n = 100, mean = 0, sd = 1)
#' Y = rnorm(n = 100, mean = X^2, sd = 1)
#' data = cbind(X, Y)
#' SPlitIndices = SPlit(data, tolerance = 1e-6, nThreads = 2) 
#' dataTest = data[SPlitIndices, ]
#' dataTrain = data[-SPlitIndices, ]
#' plot(data, main = "SPlit testing set")
#' points(dataTest, col = 'green', cex = 2)
#'
#' ## 2. An 80-20 split of the iris dataset
#' SPlitIndices = SPlit(iris, nThreads = 2)
#' irisTest = iris[SPlitIndices, ]
#' irisTrain = iris[-SPlitIndices, ]
#'
#' @references
#' Joseph, V. R., & Vakayil, A. (2020). SPlit: An Optimal Method for Data Splitting. arXiv preprint arXiv:2012.10945.
#'
#' Mak, S., & Joseph, V. R. (2018). Support points. The Annals of Statistics, 46(6A), 2562-2592.


SPlit = function(data, splitRatio=0.2, maxIterations=500, tolerance=1e-10, nThreads)
{
	if(anyNA(data))
	{
		stop("Dataset contains missing value(s).")
	}

	if(splitRatio <= 0 | splitRatio >= 1)
	{
		stop("splitRatio should be in (0, 1).")
	}

	data_ = matrix(, nrow=nrow(data))
	for(j in 1:ncol(data))
	{
		if(is.factor(data[, j]))
		{
			factor = unique(data[, j])
			factor_helm = contr.helmert(length(factor))
			data_helm = factor_helm[match(data[, j], factor), ]
			data_ = cbind(data_, data_helm)
		}
		else
		{
			if(is.numeric(data[, j]))
			{
				data_ = cbind(data_, data[, j])
			}
			else
			{
				stop("Dataset constains non-numeric non-factor column(s).")
			}
		}
	}

	data_ = data_[, -1]
	data_ = scale(data_)
	n = round(min(splitRatio, 1 - splitRatio) * nrow(data_))
	sp_ = sp(n, ncol(data_), dist.samp=data_, iter.max=maxIterations, tol=tolerance, nThreads=nThreads)
	return(subsample(data_, sp_))
}





















