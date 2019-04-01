# MultiModels

### Function

- Training machine learning problems with different models
- Ability to choose either classification or regression problems
- Models tuned to default hyper-parameters  


### Requirements

- Python 2.x/ 3.x
- The scikit-learn module

### Usages

    mmodel
     |Multimodels
        |load
        |check
        |compare
        
#### Importing Class

    from mmodel import MultiModels

#### Class MultiModels

##### Arguments

- *n_models*: Number of models to use

     Usage
     
     
     M = MultiModels(n_models=2) 
     
   **This uses '2' models**
     
- **typeof**: To choose if it's a classification problem or regression problem.

    **Usage**
    
    'clf': for classification
    
    'regr': for regression
    
    **Example**
    
    
    M = MultiModels(n_models=2, typeof='clf)
    

#### Methods

##### load(X_train, X_test, y_train, y_test)

###### Arguments

- X_train: Training data
- X_test: Testing data
- y_train: Training labels
- y_test: Testing labels 


**Returns a tuple of two dictionaries;**


- index 0: Training accuracy score 
- index 1: Test accuracy score

***

##### check(accuracy_scores)

###### Arguments

- accuracy_score: A tuple containing both training and testing accuracy scores

**Hint**: It accepts *M.load()* returns

**Prints Maximum training and testing accuracy scores**

***

##### compare(accuracy_scores)

###### Arguments

- accuracy_score: A tuple containing both training and testing accuracy scores

**Hint**: It accepts *M.load()* returns

**Prints accuracy score of all models in a fancy way(You'll like it!)** 






 




    
   
    
    
 
     
     
     


    
