from src.models.base import BaseModel
from src.models.svm_model import SVMmodel
from src.models.mlp_model import MLPmodel
from src.config import (SVM_GRID,
                        SVM_GRIDSEARCH_SUBSAMPLE,
                        SEMILLA,
                        MODELS
                        )

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

def entrenar_modelos(datos : dict,
                     semilla : int) -> dict[str, object] :
    
    dummy = DummyClassifier(strategy = 'stratified', random_state = semilla)
    dummy.fit(datos["caracteristicas_train"], datos["objetivo_train"])

    logreg = LogisticRegression(class_weight = 'balanced', max_iter = 1000, random_state = semilla)
    logreg.fit(datos["caracteristicas_train"], datos["objetivo_train"])

    svm_model : BaseModel = SVMmodel({"SVM_GRID" : SVM_GRID, 
                                       "tam_subsampleo" : SVM_GRIDSEARCH_SUBSAMPLE, 
                                       "semilla" : SEMILLA})
    
    svm_model.fit(datos["caracteristicas_train"], datos["objetivo_train"])

    mlp_model : BaseModel = MLPmodel(n_features = datos["n_features"])
    mlp_model.fit(datos["caracteristicas_train"], datos["objetivo_train"],
                  datos["caracteristicas_evaluar"], datos["objetivo_evaluar"])
    mlp_model.save( MODELS / "mlp_model.keras")

    return {
        "DUMMY" : dummy,
        "LogReg" : logreg,
        "SVM" : svm_model,
        "MLP" : mlp_model
    }