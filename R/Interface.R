L0Learn.fit <- function(X,y, Loss="SquaredError", Penalty="L0", Algorithm="CD", MaxSuppSize=100, NLambda=100, NGamma=10,
						GammaMax=10, GammaMin=0.0001, PartialSort = TRUE, MaxIters=200,
						Tol=1e-6, ActiveSet=TRUE, ActiveSetNum=3, MaxSwaps=100, ScaleDownFactor=0.8, ScreenSize=1000)
{
	G <- .Call('_L0Learn_L0LearnFit', PACKAGE = 'L0Learn', X,y, Loss, Penalty, Algorithm, MaxSuppSize, NLambda, NGamma,
						GammaMax, GammaMin, PartialSort, MaxIters, Tol, ActiveSet, ActiveSetNum, MaxSwaps, ScaleDownFactor, ScreenSize)
	class(G) <- "L0Learn"
	G$.n <- dim(X)[1]
	G$.p <- dim(X)[2]
	G
}

L0Learn.coef <- function(fit,index){
	p = fit$.p
	B = sparseVector(unlist(fit["BetaValues"][[1]][index]), unlist(fit["BetaIndices"][[1]][index]),p)
	B = as(B,"sparseMatrix")
	intercept = fit["Intercept"][[1]][index]
	#print(list(summary(B),intercept))
	out = list(Beta=B,Intercept=intercept);
	class(out) <- "L0Learncoef"
	out
}

print.L0Learncoef <- function(coef){
	print(list(Beta=summary(coef$Beta),Intercept=coef$Intercept))
}

L0Learn.predict <- function(fit,x,index){
	c = L0Learn.coef(fit,index)
	c$Intercept + x%*%c$Beta
}

print.L0Learn <- function(G){
	if(exists("Gamma",G)){
		data.frame(G["Lambda"], G["Gamma"],G["SuppSize"])
	}
	else{
		data.frame(G["Lambda"],G["SuppSize"])
	}
}
