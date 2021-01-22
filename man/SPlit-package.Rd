\name{SPlit-package}
\alias{SPlit-package}
\docType{package}
\title{SPlit}
\description{
    Split a dataset for training and testing
}
\details{
    The package 'SPlit' provides the function \code{SPlit()} to optimally split a dataset for training and testing using the method of support points (Mak and Joseph, 2018). Support points is a model-independent method for finding optimal representative points of a distribution. \code{SPlit()} attempts to obtain a split in which the distribution of both the training and testing sets resemble the distribution of the dataset. The benefits of 'SPlit' over existing data splitting procedures are detailed in Joseph and Vakayil (2020).
}
\author{
Akhil Vakayil, V. Roshan Joseph, Simon Mak

Maintainer: Akhil Vakayil <akhilv@gatech.edu>
}
\references{
    Joseph, V. R., & Vakayil, A. (2020). SPlit: An Optimal Method for Data Splitting. arXiv preprint arXiv:2012.10945.

    Mak, S., & Joseph, V. R. (2018). Support points. The Annals of Statistics, 46(6A), 2562-2592.
}
\keyword{ package }
