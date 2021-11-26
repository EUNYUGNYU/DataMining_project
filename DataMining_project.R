# 실행시간 측정
time1<-Sys.time()
time1

# 패키지 불러오기
library(tidymodels)
library(caret)
library(skimr)
library(MASS)
library(gridExtra)
library(scales)
library(naniar)
library(lattice)
library(partykit)
library(rpart.plot)

# 데이터 불러오기
DF<-as.data.frame(read.csv("GSS2018B.csv"))
dim(DF)
str(DF)

# 변수조정
DF<-dplyr::select(DF,-PRES12)
DF<-mutate(DF,
           PRES16=factor(ifelse(PRES16=='T','N','Y')),
           RACE=factor(RACE),
           SEX=factor(SEX),
           MARITAL=factor(MARITAL))

DF<-mutate(DF,
           RACE=factor(RACE, labels=make.names(levels(RACE))),
           SEX=factor(SEX,labels=make.names(levels(SEX))),
           MARITAL=factor(MARITAL,labels=make.names(levels(MARITAL))))
           
head(DF)
summary(DF)

# 결측치 파악
DF %>% skim()
DF %>%group_by(PRES16) %>% skim()

sum(complete.cases(DF))/nrow(DF)*100 #전체 자료의 89.7%의 자료만 사용함
naniar::vis_miss(DF)
naniar::miss_var_summary(DF)


# TR, TS 분할
set.seed(20171018)
Ris<-initial_split(DF, prop=0.75, strata=PRES16)
TR<-training(Ris)
TS<-testing(Ris)


# 전처리 객체 만들기
RC <-
  recipe(PRES16~., data=TR) %>% 
  step_medianimpute(all_numeric(),-all_outcomes()) %>% 
  step_modeimpute(all_nominal(),-all_outcomes()) %>% 
  step_dummy(all_nominal(),-all_outcomes())
RC

# 튜닝계획 지정
trCntl<-trainControl(method='cv', number=10,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE)



# 1. glm

# 튜닝 모수 확인
modelLookup('glm')

# 적합
set.seed(20171018)
Mglm<-train(RC, data=TR,
           method='glm',
           family='binomial',
           metric='ROC',
           trControl = trCntl)
Mglm

Mglm$results
summary(Mglm)
ggplot(varImp(Mglm))
Mglm$bestTune
Mglm$finalModel
Mglm$resample

# 예측값 저장
TROUT<-TR %>% dplyr::select(PRES16)
TSOUT<-TS %>% dplyr::select(PRES16)
TROUT<-TROUT %>% bind_cols(phglm=predict(Mglm, newdata=TR, type='prob')[,'Y'])
TSOUT<-TSOUT %>% bind_cols(phglm=predict(Mglm, newdata=TS, type='prob')[,'Y'])
TROUT<-TROUT %>% bind_cols(yhglm=predict(Mglm, newdata=TR))
TSOUT<-TSOUT %>% bind_cols(yhglm=predict(Mglm, newdata=TS))
head(TSOUT)

# TS에서의 성능평가

foo <- function(y, ph, yh, event_level='first'){
  
  cnf <- table(yh, y)
  if(event_level =='second'){ 
    tn <- cnf[1,1]; fn <- cnf[1,2]; fp <- cnf [2,1]; tp <- cnf[2,2];
  }else { 
    tn <-cnf [2,2]; fn <- cnf[2,1]; fp <- cnf[1,2]; tp <- cnf[1,1];
  }
    
  c(acc = accuracy_vec(y, yh), 
    auc = roc_auc_vec(y, ph, event_level=event_level), 
    prauc = pr_auc_vec(y, ph, event_level=event_level), 
    f1 = f_meas_vec(y, yh, event_level=event_level), 
    kap = kap_vec(y, yh), 
    sens= sens_vec(y, yh, event_level=event_level), 
    spec = spec_vec(y, yh, event_level=event_level), 
    prec = precision_vec(y, yh, event_level=event_level), 
    tn = tn, 
    fn = fn, 
    fp = fp, 
    tp = tp
  )
}

foo(TSOUT$PRES16, TSOUT$phglm, TSOUT$yhglm, event_level='second') 

# 잔차 그래프
g1 <- autoplot(roc_curve (TROUT, 'PRES16', 'phglm', event_level='second')) 
g2 <- autoplot(roc_curve(TSOUT, 'PRES16', 'phglm', event_level='second')) 
g3 <- autoplot(pr_curve (TROUT, 'PRES16', 'phglm', event_level='second')) 
g4 <- autoplot(pr_curve (TSOUT, 'PRES16', 'phglm', event_level='second')) 
grid.arrange(g1,g2,g3,g4, ncol=2) 

confusionMatrix(data=TSOUT$yhglm, reference=TSOUT$PRES16, positive='Y')

cnf<-conf_mat(TSOUT,'PRES16','yhglm')
summary(cnf, event_level='second')
autoplot(cnf, event_level='second')

METglm<-
  bind_cols(
    bind_rows(foo(TROUT$PRES16, TROUT$phglm, TROUT$yhglm, event_level = 'second'),
              foo(TSOUT$PRES16, TSOUT$phglm, TSOUT$yhglm, event_level = 'second')),
    data.frame(model='glm', TRTS=c('TR','TS')))
METglm


# 2. glmStepAIC  : AIC 변수 선택

# 튜닝 모수 확인
modelLookup('glmStepAIC')

# 적합
set.seed(20171018)
Mstep<-train(RC, data=TR,
             method='glmStepAIC',
             direction='backward',
             metric='ROC',
             trControl = trCntl)
Mstep

Mstep$results
summary(Mstep)
ggplot(varImp(Mstep))
Mstep$bestTune
Mstep$finalModel
Mstep$resample

# 예측값 저장
TROUT<-TROUT %>% bind_cols(phstep=predict(Mstep, newdata=TR, type='prob')[,'Y'])
TSOUT<-TSOUT %>% bind_cols(phstep=predict(Mstep, newdata=TS, type='prob')[,'Y'])
TROUT<-TROUT %>% bind_cols(yhstep=predict(Mstep, newdata=TR))
TSOUT<-TSOUT %>% bind_cols(yhstep=predict(Mstep, newdata=TS))
head(TSOUT)

# TS에서의 성능평가
foo(TSOUT$PRES16, TSOUT$phstep, TSOUT$yhstep, event_level='second') 

# 잔차 그래프
g1 <- autoplot(roc_curve (TROUT, 'PRES16', 'phstep', event_level='second')) 
g2 <- autoplot(roc_curve(TSOUT, 'PRES16', 'phstep', event_level='second')) 
g3 <- autoplot(pr_curve (TROUT, 'PRES16', 'phstep', event_level='second')) 
g4 <- autoplot(pr_curve (TSOUT, 'PRES16', 'phstep', event_level='second')) 
grid.arrange(g1,g2,g3,g4, ncol=2) 

confusionMatrix(data=TSOUT$yhstep, reference=TSOUT$PRES16, positive='Y')

cnf<-conf_mat(TSOUT,'PRES16','yhstep')
summary(cnf, event_level='second')
autoplot(cnf, event_level='second')

METstep<-
  bind_cols(
    bind_rows(foo(TROUT$PRES16, TROUT$phstep, TROUT$yhstep, event_level = 'second'),
              foo(TSOUT$PRES16, TSOUT$phstep, TSOUT$yhstep, event_level = 'second')),
    data.frame(model='glmStepAIC', TRTS=c('TR','TS')))
METstep


# 3. glmnet

modelLookup('glmnet')

# 적합
set.seed(20171018)
glmnetGrid<-expand.grid(alpha=seq(0,1,by=0.25), lambda=seq(0.0, 0.1, by=0.01))
Mglmnet<-train(RC, data=TR,
               method='glmnet',
               family='binomial',
               metric = 'ROC',
               trControl = trCntl,
               tuneGrid = glmnetGrid)
Mglmnet

Mglmnet$results
ggplot(Mglmnet)
ggplot(varImp(Mglmnet))
Mglmnet$bestTune
plot(Mglmnet$finalModel) 


# 예측값 저장
TROUT<-TROUT %>% bind_cols(phglmnet=predict(Mglmnet, newdata=TR, type='prob')[,'Y'])
TSOUT<-TSOUT %>% bind_cols(phglmnet=predict(Mglmnet, newdata=TS, type='prob')[,'Y'])
TROUT<-TROUT %>% bind_cols(yhglmnet=predict(Mglmnet, newdata=TR))
TSOUT<-TSOUT %>% bind_cols(yhglmnet=predict(Mglmnet, newdata=TS))

# TS에서의 성능평가
foo(TSOUT$PRES16, TSOUT$phglmnet, TSOUT$yhglmnet, event_level='second') 

# 잔차 그래프
g1 <- autoplot(roc_curve (TROUT, 'PRES16', 'phglmnet', event_level='second')) 
g2 <- autoplot(roc_curve(TSOUT, 'PRES16', 'phglmnet', event_level='second')) 
g3 <- autoplot(pr_curve (TROUT, 'PRES16', 'phglmnet', event_level='second')) 
g4 <- autoplot(pr_curve (TSOUT, 'PRES16', 'phglmnet', event_level='second')) 
grid.arrange(g1,g2,g3,g4, ncol=2) 

confusionMatrix(data=TSOUT$yhglmnet, reference=TSOUT$PRES16, positive='Y')

cnf<-conf_mat(TSOUT,'PRES16','yhglmnet')
summary(cnf, event_level='second')
autoplot(cnf, event_level='second')



METglmnet<-
  bind_cols(
    bind_rows(foo(TROUT$PRES16, TROUT$phglmnet, TROUT$yhglmnet, event_level = 'second'),
              foo(TSOUT$PRES16, TSOUT$phglmnet, TSOUT$yhglmnet, event_level = 'second')),
    data.frame(model='glmnet', TRTS=c('TR','TS')))
METglmnet



# 4.nnet


modelLookup('nnet')

# 적합
set.seed(20171018)
nnetGrid<-expand.grid(size=2^(1:8), decay=seq(0.0,0.1,by=0.025))
Mnnet<-train(RC, data=TR,
             method='nnet',
             maxit=1000,trace=F,
             metric = 'ROC',
             trControl = trCntl,
             tuneGrid = nnetGrid)
Mnnet

Mnnet$results
ggplot(Mnnet)
ggplot(varImp(Mnnet))
Mnnet$bestTune
summary(Mnnet$finalModel) 
Mnnet$resample

# 예측값 저장
TROUT<-TROUT %>% bind_cols(phnnet=predict(Mnnet, newdata=TR, type='prob')[,'Y'])
TSOUT<-TSOUT %>% bind_cols(phnnet=predict(Mnnet, newdata=TS, type='prob')[,'Y'])
TROUT<-TROUT %>% bind_cols(yhnnet=predict(Mnnet, newdata=TR))
TSOUT<-TSOUT %>% bind_cols(yhnnet=predict(Mnnet, newdata=TS))
head(TSOUT)

# TS에서의 성능평가
foo(TSOUT$PRES16, TSOUT$phnnet, TSOUT$yhnnet, event_level='second') 

# 잔차 그래프
g1 <- autoplot(roc_curve (TROUT, 'PRES16', 'phnnet', event_level='second')) 
g2 <- autoplot(roc_curve(TSOUT, 'PRES16', 'phnnet', event_level='second')) 
g3 <- autoplot(pr_curve (TROUT, 'PRES16', 'phnnet', event_level='second')) 
g4 <- autoplot(pr_curve (TSOUT, 'PRES16', 'phnnet', event_level='second')) 
grid.arrange(g1,g2,g3,g4, ncol=2) 

confusionMatrix(data=TSOUT$yhnnet, reference=TSOUT$PRES16, positive='Y')

cnf<-conf_mat(TSOUT,'PRES16','yhnnet')
summary(cnf, event_level='second')
autoplot(cnf, event_level='second')


# TS에서의 성능평가
METnnet<-
  bind_cols(
    bind_rows(foo(TROUT$PRES16, TROUT$phnnet, TROUT$yhnnet, event_level = 'second'),
              foo(TSOUT$PRES16, TSOUT$phnnet, TSOUT$yhnnet, event_level = 'second')),
    data.frame(model='nnet', TRTS=c('TR','TS')))
METnnet

# svmRadial


modelLookup('svmRadial')

# 적합
set.seed(20171018)
svmRadialGrid<-expand.grid(.sigma=2^(-4:4),.C=2^(seq(-4,4)))
MsvmRadial<-train(RC, data=TR,
                  method='svmRadial',
                  metric = 'ROC',
                  trControl = trCntl,
                  tuneGrid = svmRadialGrid)
MsvmRadial

MsvmRadial$results
ggplot(MsvmRadial)
ggplot(varImp(MsvmRadial))
MsvmRadial$bestTune
summary(MsvmRadial$finalModel) 
MsvmRadial$resample

# 예측값 저장
TROUT<-TROUT %>% bind_cols(phsvmRadial=predict(MsvmRadial, newdata=TR, type='prob')[,'Y'])
TSOUT<-TSOUT %>% bind_cols(phsvmRadial=predict(MsvmRadial, newdata=TS, type='prob')[,'Y'])
TROUT<-TROUT %>% bind_cols(yhsvmRadial=predict(MsvmRadial, newdata=TR))
TSOUT<-TSOUT %>% bind_cols(yhsvmRadial=predict(MsvmRadial, newdata=TS))
head(TSOUT)

# TS에서의 성능평가
foo(TSOUT$PRES16, TSOUT$phsvmRadial, TSOUT$yhsvmRadial, event_level='second') 

# 잔차 그래프
g1 <- autoplot(roc_curve (TROUT, 'PRES16', 'phsvmRadial', event_level='second')) 
g2 <- autoplot(roc_curve(TSOUT, 'PRES16', 'phsvmRadial', event_level='second')) 
g3 <- autoplot(pr_curve (TROUT, 'PRES16', 'phsvmRadial', event_level='second')) 
g4 <- autoplot(pr_curve (TSOUT, 'PRES16', 'phsvmRadial', event_level='second')) 
grid.arrange(g1,g2,g3,g4, ncol=2) 

confusionMatrix(data=TSOUT$yhsvmRadial, reference=TSOUT$PRES16, positive='Y')

cnf<-conf_mat(TSOUT,'PRES16','yhsvmRadial')
summary(cnf, event_level='second')
autoplot(cnf, event_level='second')


# TS에서의 성능평가
METsvmRadial<-
  bind_cols(
    bind_rows(foo(TROUT$PRES16, TROUT$phsvmRadial, TROUT$yhsvmRadial, event_level = 'second'),
              foo(TSOUT$PRES16, TSOUT$phsvmRadial, TSOUT$yhsvmRadial, event_level = 'second')),
    data.frame(model='svmRadial', TRTS=c('TR','TS')))
METsvmRadial


# rpart
modelLookup('rpart')
modelLookup('rpart2')
# 적합
set.seed(20171018)
rpartGrid<-expand.grid(cp=seq(0,0.1, length=10))
Mrpart<-train(RC, data=TR,
              method='rpart',
              metric = 'ROC',
              trControl = trCntl,
              tuneGrid = rpartGrid)
Mrpart

Mrpart$results
ggplot(Mrpart)
ggplot(varImp(Mrpart))
Mrpart$bestTune

library(partykit)
library(rpart.plot)
plot(as.party(Mrpart$finalModel))
rpart.plot::rpart.plot(Mrpart$finalModel)

# 예측값 저장
TROUT<-TROUT %>% bind_cols(phrpart=predict(Mrpart, newdata=TR, type='prob')[,'Y'])
TSOUT<-TSOUT %>% bind_cols(phrpart=predict(Mrpart, newdata=TS, type='prob')[,'Y'])
TROUT<-TROUT %>% bind_cols(yhrpart=predict(Mrpart, newdata=TR))
TSOUT<-TSOUT %>% bind_cols(yhrpart=predict(Mrpart, newdata=TS))
head(TSOUT)

# TS에서의 성능평가
foo(TSOUT$PRES16, TSOUT$phrpart, TSOUT$yhrpart, event_level='second') 

# 잔차 그래프
g1 <- autoplot(roc_curve (TROUT, 'PRES16', 'phrpart', event_level='second')) 
g2 <- autoplot(roc_curve(TSOUT, 'PRES16', 'phrpart', event_level='second')) 
g3 <- autoplot(pr_curve (TROUT, 'PRES16', 'phrpart', event_level='second')) 
g4 <- autoplot(pr_curve (TSOUT, 'PRES16', 'phrpart', event_level='second')) 
grid.arrange(g1,g2,g3,g4, ncol=2) 

confusionMatrix(data=TSOUT$yhrpart, reference=TSOUT$PRES16, positive='Y')

cnf<-conf_mat(TSOUT,'PRES16','yhrpart')
summary(cnf, event_level='second')
autoplot(cnf, event_level='second')


# TS에서의 성능평가
METrpart<-
  bind_cols(
    bind_rows(foo(TROUT$PRES16, TROUT$phrpart, TROUT$yhrpart, event_level = 'second'),
              foo(TSOUT$PRES16, TSOUT$phrpart, TSOUT$yhrpart, event_level = 'second')),
    data.frame(model='rpart', TRTS=c('TR','TS')))
METrpart


# ranger

modelLookup('ranger')

# 적합
set.seed(20171018)
rangerGrid<-expand.grid(
  mtry=seq(2,ncol(TR)-1, by=2),
  splitrule=c('gini','extratrees'),
  min.node.size=1:3)

Mranger<-train(RC, data=TR,
               method='ranger',
               importance='impurity',
               metric = 'ROC',
               trControl = trCntl,
               tuneGrid = rangerGrid)
Mranger

Mranger$results
ggplot(Mranger)
ggplot(varImp(Mranger))
Mranger$bestTune
Mranger$finalModel
Mranger$resample

# 예측값 저장
TROUT<-TROUT %>% bind_cols(phranger=predict(Mranger, newdata=TR, type='prob')[,'Y'])
TSOUT<-TSOUT %>% bind_cols(phranger=predict(Mranger, newdata=TS, type='prob')[,'Y'])
TROUT<-TROUT %>% bind_cols(yhranger=predict(Mranger, newdata=TR))
TSOUT<-TSOUT %>% bind_cols(yhranger=predict(Mranger, newdata=TS))
head(TSOUT)

# TS에서의 성능평가
foo(TSOUT$PRES16, TSOUT$phranger, TSOUT$yhranger, event_level='second') 

# 잔차 그래프
g1 <- autoplot(roc_curve (TROUT, 'PRES16', 'phranger', event_level='second')) 
g2 <- autoplot(roc_curve(TSOUT, 'PRES16', 'phranger', event_level='second')) 
g3 <- autoplot(pr_curve (TROUT, 'PRES16', 'phranger', event_level='second')) 
g4 <- autoplot(pr_curve (TSOUT, 'PRES16', 'phranger', event_level='second')) 
grid.arrange(g1,g2,g3,g4, ncol=2) 

confusionMatrix(data=TSOUT$yhranger, reference=TSOUT$PRES16, positive='Y')

cnf<-conf_mat(TSOUT,'PRES16','yhranger')
summary(cnf, event_level='second')
autoplot(cnf, event_level='second')


# TS에서의 성능평가
METranger<-
  bind_cols(
    bind_rows(foo(TROUT$PRES16, TROUT$phranger, TROUT$yhranger, event_level = 'second'),
              foo(TSOUT$PRES16, TSOUT$phranger, TSOUT$yhranger, event_level = 'second')),
    data.frame(model='rager', TRTS=c('TR','TS')))
METranger

# gbm
modelLookup('gbm')

# 적합
set.seed(20171018)
gbmGrid<-expand.grid(
  n.trees=c(100,200,500),
  interaction.depth=c(3,5,10),
  shrinkage=0.1,
  n.minobsinnode=10
)

Mgbm<-train(RC, data=TR,
            method='gbm',
            metric = 'ROC',
            trControl = trCntl,
            tuneGrid = gbmGrid)
Mgbm

Mgbm$results
ggplot(Mgbm)
ggplot(varImp(Mgbm))
Mgbm$bestTune
Mgbm$finalModel
Mgbm$resample

# 예측값 저장
TROUT<-TROUT %>% bind_cols(phgbm=predict(Mgbm, newdata=TR, type='prob')[,'Y'])
TSOUT<-TSOUT %>% bind_cols(phgbm=predict(Mgbm, newdata=TS, type='prob')[,'Y'])
TROUT<-TROUT %>% bind_cols(yhgbm=predict(Mgbm, newdata=TR))
TSOUT<-TSOUT %>% bind_cols(yhgbm=predict(Mgbm, newdata=TS))
head(TSOUT)

# TS에서의 성능평가
foo(TSOUT$PRES16, TSOUT$phgbm, TSOUT$yhgbm, event_level='second') 

# 잔차 그래프
g1 <- autoplot(roc_curve (TROUT, 'PRES16', 'phgbm', event_level='second')) 
g2 <- autoplot(roc_curve(TSOUT, 'PRES16', 'phgbm', event_level='second')) 
g3 <- autoplot(pr_curve (TROUT, 'PRES16', 'phgbm', event_level='second')) 
g4 <- autoplot(pr_curve (TSOUT, 'PRES16', 'phgbm', event_level='second')) 
grid.arrange(g1,g2,g3,g4, ncol=2) 

confusionMatrix(data=TSOUT$yhgbm, reference=TSOUT$PRES16, positive='Y')

cnf<-conf_mat(TSOUT,'PRES16','yhgbm')
summary(cnf, event_level='second')
autoplot(cnf, event_level='second')


# TS에서의 성능평가
METgbm<-
  bind_cols(
    bind_rows(foo(TROUT$PRES16, TROUT$phgbm, TROUT$yhgbm, event_level = 'second'),
              foo(TSOUT$PRES16, TSOUT$phgbm, TSOUT$yhgbm, event_level = 'second')),
    data.frame(model='gbm', TRTS=c('TR','TS')))
METgbm


#xgb

modelLookup('xgbTree')

# 적합
set.seed(20171018)
xgbGrid<-expand.grid(
  nrounds=c(1,10),
  max_depth=c(1,4),
  eta=c(.1,.4),
  gamma=0,
  colsample_bytree=.7,
  min_child_weight=1,
  subsample=c(.8,1))


Mxgb<-train(RC, data=TR,
            method='xgbTree',
            metric = 'ROC',
            trControl = trCntl,
            tuneGrid = xgbGrid)
Mxgb

Mxgb$results
ggplot(Mxgb)
ggplot(varImp(Mxgb))
Mxgb$bestTune
Mxgb$finalModel
Mxgb$resample

# 예측값 저장
TROUT<-TROUT %>% bind_cols(phxgb=predict(Mxgb, newdata=TR, type='prob')[,'Y'])
TSOUT<-TSOUT %>% bind_cols(phxgb=predict(Mxgb, newdata=TS, type='prob')[,'Y'])
TROUT<-TROUT %>% bind_cols(yhxgb=predict(Mxgb, newdata=TR))
TSOUT<-TSOUT %>% bind_cols(yhxgb=predict(Mxgb, newdata=TS))
head(TSOUT)

# TS에서의 성능평가
foo(TSOUT$PRES16, TSOUT$phxgb, TSOUT$yhxgb, event_level='second') 

# 잔차 그래프
g1 <- autoplot(roc_curve (TROUT, 'PRES16', 'phxgb', event_level='second')) 
g2 <- autoplot(roc_curve(TSOUT, 'PRES16', 'phxgb', event_level='second')) 
g3 <- autoplot(pr_curve (TROUT, 'PRES16', 'phxgb', event_level='second')) 
g4 <- autoplot(pr_curve (TSOUT, 'PRES16', 'phxgb', event_level='second')) 
grid.arrange(g1,g2,g3,g4, ncol=2) 

confusionMatrix(data=TSOUT$yhxgb, reference=TSOUT$PRES16, positive='Y')

cnf<-conf_mat(TSOUT,'PRES16','yhxgb')
summary(cnf, event_level='second')
autoplot(cnf, event_level='second')


# TS에서의 성능평가
METxgb<-
  bind_cols(
    bind_rows(foo(TROUT$PRES16, TROUT$phxgb, TROUT$yhxgb, event_level = 'second'),
              foo(TSOUT$PRES16, TSOUT$phxgb, TSOUT$yhxgb, event_level = 'second')),
    data.frame(model='xgbTree', TRTS=c('TR','TS')))
METxgb


# CV 성능평가

RESAMP<-resamples(list(GLM=Mglm,
                       STEP=Mstep,
                       GLMNET=Mglmnet,
                       NNET=Mnnet,
                       SVMR=MsvmRadial,
                       RPART=Mrpart,
                       RANGER=Mranger,
                       GBM=Mgbm,
                       XGB=Mxgb))
summary(RESAMP)

bwplot(RESAMP)


# TR/ TS 평가
MET<-bind_rows(METglm, METstep,METglmnet,METnnet, METsvmRadial,
               METrpart,METranger,METgbm,METxgb)
MET<-arrange(
  bind_rows(METglm, METstep,METglmnet,METnnet, METsvmRadial,
            METrpart,METranger,METgbm,METxgb),desc(auc))


g1<-ggplot(MET, aes(x=model, y=acc, shape=TRTS, col=TRTS, group=TRTS))+
  geom_line()+
  geom_point(size=3)
g2<-ggplot(MET, aes(x=model, y=auc, shape=TRTS, col=TRTS, group=TRTS))+
  geom_line()+
  geom_point(size=3)
grid.arrange(g1, g2, nrow=2, ncol=1)

MET %>% 
  dplyr::select(auc, model,TRTS) %>% 
  filter(TRTS=="TS")

#최종모형 선택
yhsvmRadial=predict(MsvmRadial, newdata=TS)
phsvmRadial=predict(MsvmRadial, newdata=TS, type='prob')[,'Y']
final<-cbind(TS,yhsvmRadial,phsvmRadial)
final %>% 
  head(10)

final %>% 
  arrange(phsvmRadial) %>% 
  head(5)
final %>% 
  arrange(-phsvmRadial) %>% 
  head(5)


predict(MsvmRadial, newdata=TS, type='prob') %>% 
  arrange(Y) %>% 
  head(3)
TSOUT%>% 
  dplyr::select(PRES16, yhnnet,phsvmRadial)
time2<-Sys.time()
time2
time2-time1
