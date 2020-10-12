library("L0Learn")

expect_equal_cv <- function(x, y, cv_tolerance=1e-6, info=NULL) {
    for (i in seq_along(x)){
        if (startsWith(names(x)[i], 'cv')){
            expect_equal(x[i], y[i], tolerance=cv_tolerance, info=info)
        } else {
            expect_equal(x[i], y[i], info=info)
        }
    }
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