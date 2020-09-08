library("L0Learn")
# 
# check_similar_fit_solution <- function(x, y, tolerance=1e-5, cv_tolerance=1e-3){
#     for (s in c("p", "n", "varnames", "settings", "loss", "penalty", "gamma")){
#         expect_identical(x[[s]], y[[s]])
#     }
#     
#     # This is not an ideal test. Somehow, even though I set the seed before each
#     # test there is a difference in solution size. Indictating a bug in the Rcpp
#     # code.
#     
#     num_s = min(x[["beta"]][[1]]@Dim[[2]],
#                 y[["beta"]][[1]]@Dim[[2]])
#     
#     for (s in c("lambda", "a0", "converged", "suppSize")){
#         expect_equal(x[[s]][[1]][1:num_s],
#                      y[[s]][[1]][1:num_s],
#                      tolerance=tolerance)
#     }
#     
#     expect_equal(as.matrix(x[["beta"]][[1]])[, 1:num_s],
#                  as.matrix(y[["beta"]][[1]])[, 1:num_s],
#                  tolerance=tolerance)
#     
# }

expect_equal_cv <- function(x, y, cv_tolerance=1e-6) {
    for (i in seq_along(x)){
        if (startsWith(names(x)[i], 'cv')){
            expect_equal(x[i], y[i], tolerance=cv_tolerance)
        } else {
            expect_equal(x[i], y[i])
        }
    }
}


funcFromStr <- function(func){
    if (func == 'fit'){
        func <- L0Learn.fit
    } else if (func == 'cvfit'){
        func <- L0Learn.cvfit
    } else {
        fail("Unkown function")
    }
    func
}


savetest <- function(seed, func, x, y, params, name){
    func_ <- funcFromStr(func)
    set.seed(seed)
    
    result <- do.call(func_, c(list(x=x, y=y), params))
    
    savedResult <- c(list(func=func, x=x, y=y), list(params=params), list(result=result))
    save(savedResult, file=name)
    NULL
}


runtest <- function(testlist){
    set.seed(testlist$seed)
    
    func <- funcFromStr(testlist$func)
    
    result <- do.call(func, c(list(testlist$x, testlist$y), testlist$params))
    
    expect_equal_cv(result, testlist$result)
}