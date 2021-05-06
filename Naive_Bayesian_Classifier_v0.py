#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu March 17 10:04:31 2021
@author: Thomaths QuantX
"""

"""
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)  https://creativecommons.org/licenses/by-nc-sa/4.0/
Under the following terms: 
Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. 
            You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
NonCommercial — You may not use the material for commercial purposes.
ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
"""

'''
Palmer's penguins - credit \href{https://allisonhorst.github.io/palmerpenguins/articles/intro.html}{Alison Horst} 
'''


import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go


## --------------- quelques fonctions pour la gestion des fichiers/dossiers de destinations -------------------------
'''
Ces fonctions sont utiles pour gérer les fichiers en sortie du classifier 
(une fois la partie sur l'optimisation des amplitudes effectuée). 
Elles peuvent cependant faire l'objet d'une librairie personnelle : retour sur la modularité
'''

import os

def erreur_check_file(chemin_fichier):
    '''
    return : existence : bool, chemin : str, erreur : bool
    '''
    if os.path.exists(chemin_fichier):        
        Erreur_ = False
    else:  
        Erreur_ = True
    return Erreur_, chemin_fichier

def suppresion_(fichier):
    try:
        os.remove(fichier)
        return False
    except:
        print("erreur dans la suppression")
        return True

def creation_dossier(chemin_dossier):
    if not os.path.exists(chemin_dossier):
        os.makedirs(chemin_dossier)
    return chemin_dossier


## ------------------ concernant le tri rapide -------------------------
    
#une proposition récursive non en place : 

def tri_rapide(tableau):
    if tableau == []:
        return []
    else:
        pivot = tableau[-1]
        valeurs_inferieures = [x for x in tableau if x < pivot]
        valeurs_superieures = [x for x in tableau[:-1] if x >= pivot]
        return tri_rapide(valeurs_inferieures) + [pivot] + tri_rapide(valeurs_superieures)
     
# tableau = [2,5,4,3,7,1,9,9,6,8,-1]
# L = tri_rapide(tableau)
# print(L)
    
    
#une proposition itérative : 
        
      # .... pour la suite ....



## ------------------ concernant le classifier en tant que tel -------------------------   
        
'''
Choix d'utiliser une classe car plus maléable avec des méthodes d'un seul objet,  permet de stocker dans ses 
'''   
    
class NaiveBayesClassifier():
    
    def __init__(self,donnees_entrainement : list, donnees_de_test : list, libelles : list, reference : str, amplitudes : list = None, selection_caracteristiques : list = None) :
        '''
        constructeur du classifier
        input : 
            - les jeux de données (donnes_entrainement, donnees_de_test), 
            - les libelles et si des colonnes spécifiques sont sélectionnées (selection_caracteristiques, None par défaut)
            - référence : la catégorie des objets que l'on regarde (Palmer_penguin, pokemon, spams, ...) 
            - amplitudes : possiblement non renseignées, sinon des valeurs spécifiques pour avoir 
        '''
        
        # ---- quelques assertions pour vérifier un minimum ----------
        assert type(donnees_entrainement) is list, "erreur dans les donnees d'entraînement"
        assert donnees_entrainement != [] or donnees_entrainement is not None, "pas de données d'entraînement"
        assert donnees_de_test != [] or donnees_de_test is None, "pas de données de test"        
        assert type(donnees_de_test) is list, "erreur dans les donnees de test"
        assert type(libelles) is list, "erreur dans la liste des libellés"        
        
        self.dataset = donnees_entrainement
        self.nombre_de_donnees_entrainement = len(donnees_entrainement) #permet de calculer la probabilité d'un label et de vérifier que toutes les données sont prises en compte
        self.datatest = donnees_de_test
        self.labels =  None
        self.libelles = libelles 
        self.reference = reference  #pour le nom des fichiers
        self.dictionnaire_probas = { }    #contiendra toutes les informations sur les distributions de probabilités et quelques variables utiles
        self.bins = amplitudes #si on propose des amplitudes pour les caractéristiques 
        if selection_caracteristiques is None: 
            self.colonnes = [i for i in range(len(libelles)-1)]  #on sélectionne toutes les caractéristiques.  len()-1 car la dernière colonne est le label
        else:
            assert type(selection_caracteristiques) is list, "erreur dans la liste des colonnes sélectionnées"
            self.colonnes = selection_caracteristiques


    def __str__(self):
        '''
        pour afficher quelques valeurs mais surtout utile au début de l'élaboration de l'algorithme, peu utile ensuite
        '''
        return str(self.reference)+str(self.libelles)+str(self.colonnes)+str(self.labels)
    
    
    def etude_des_labels_et_repartition_donnees(self):        
        '''
        methode ayant pour but d'analyser la derniere colonne des donnees qui contient les differents labels, et de les lister
        Cette méthode est appelée non initialement mais quand on lance l'apprentissage (si jamais, on peut mieux articuler).
        Elle permet aussi de trier les données suivant leur label, comme on regarde donnée par donnée (gain de temps)
        
        input : None
        output : pas vraiment en sortie, mais actualisation des labels, et surtout, une liste des données des différents labels triées par label
        
        on remplace les données d'entraînement par une autre liste de listes, mais ordonnées
        
        '''
        
        liste_des_labels = [ ]        
        donnees_par_label = [ ]
        
        for element in self.dataset:
            if not element[-1] in liste_des_labels: #si le label n'est pas déjà listé
                liste_des_labels.append(element[-1])
                donnees_par_label.append([element])
            else:
                n = 0                                 #sinon, on ajoute les données à la sous-liste associée au label
                bon_label = False
                while n<len(liste_des_labels) and not bon_label: #on se met dans la liste au niveau de la sous-liste associée au label
                    if element[-1]==liste_des_labels[n]:
                        bon_label = True
                        donnees_par_label[n].append(element)                         
                    n += 1 
                                  
        self.labels = liste_des_labels #initialement None maintenant contient la liste des labels
        self.dataset = donnees_par_label #on ecrase le lien qui pointe vers une nouvelle liste

                          
    
    def preconstruction_dictionnaire_probas(self):
        
        '''
        Le dictionnaire contenant toutes les distributions et probabilités est initialement vide. 
        On va le remplir avec  (si labels sont "A", "B", "C", ...)
         - clé "labels" : contient un dictionnaire des probabilités de chaque occurence de label  
             dictionnaire_probas["labels"] = {"A": 0.01, "B":0.59, "C":0.4}
         - clé pour chaque label, contient effectif du label et ici les listes des valeurs pour chaque caractéristique
             dictionnaire_probas["A"] = {"effectif":9,  "caracteristique1" = [1, 2, -2, -3, ..],  "caracteristique2" = [0.3, 0.8, 0.9, ...] } 
             dictionnaire_probas["B"] = {"effectif":12,  "caracteristique1" = [8,9,8,7],  "caracteristique2" = [0.1, 0.2, 0.3, ...] } 
             ....
        '''
        
        #---- construction et initialisation du dictionnaire des probabilités pour les labels  P(labels)
        self.dictionnaire_probas["labels"] = {}
        
        position_label = 0 #pour lier le numero de la colonne à son nom, afin de l'afficher dans le dictionnaire
        for label in self.labels:
            #determination de la probabilité du label : on est sûr de ne pas diviser par 0 car on a !=[] and is not None 
            self.dictionnaire_probas["labels"][label] = len(self.dataset[position_label])/self.nombre_de_donnees_entrainement  
            
        #print(self.dictionnaire_probas)
            
        #---- pour chaque label, pour chaque feature, on va construire une liste que l'on va trier grâce à un tri fusion, rapide , ...  
        #---- pour l'instant, ce dictionnaire ne contient les probabilités que pour les labels, pas pour leur feature
            
            self.dictionnaire_probas[label]={"effectif":len(self.dataset[position_label])}           
           
            
            for feature in self.colonnes:                       
                liste_des_valeurs_du_feature = [ ]         
                for ligne in self.dataset[position_label]:                        #pour chaque label, pour chaque caractéristique,
                    liste_des_valeurs_du_feature.append(float(ligne[feature]))    # on stocke les valeurs dans une liste  ... 
            
                self.dictionnaire_probas[label][self.libelles[feature]] = tri_rapide(liste_des_valeurs_du_feature)   # ... que l'on trie ici !! 
                    
            position_label += 1 #on augmente de 1 quand on passe au label suivant 
  

        #on sauvegarde les données dans dataset qui est inutilisé car on va tester les differentes amplitudes
        #ou directement self.dataset = self.dictionnaire_probas.copy()  sinon           
            
        self.dataset = { } 
        for element in self.dictionnaire_probas.keys():            
            self.dataset[element] = self.dictionnaire_probas[element]
            
        # for element in self.dataset.keys():
        #     print(element, "\n", self.dataset[element], "\n")



    def construction_distributions_suivant_amplitudes(self):
        '''
        objectif : connaissant les amplitudes demandée pour chaque caracteristique/feature, on va extraire l'ensemble
        des valeurs d'une caracteristique dans une variable, nommée "serie_statistique" et faire des intervalles/classes suivant 
        l'amplitude indiquée dans self.bins       
        On aura alors pour chaque label, un dictionnaire contenant effectif et distribution/histogramme des caractéristiques  (liste de listes) :        
             dictionnaire_probas["A"] = {"effectif":9,  "caracteristique1" = [[0,0], [3,3], [6,5], [9,1], [12,0]],  "caracteristique2" = similaire } 
             dictionnaire_probas["B"] = {"effectif":12,  "caracteristique1" =[[0,3], [1,8], [2,1], [3,0]] ,  "caracteristique2" = ...} 
             L = [[0,0], [3,3], [6,5], [9,1], [12,0]]: signifie que amplitude de  3 car L[i][O] étant les caractères, la différence est de 3 entre deux éléments 
             effectif de 0 dans l'intervalle [ 0 ; 3 [
             effectif de 3 dans l'intervalle [ 3 ; 6 [
             effectif de 5 dans l'intervalle [ 6 ; 9 [
             effectif de 1 dans l'intervalle [ 9 ; 12 [
             effectif de 0 dans l'intervalle [ 12; 15 [
             ....
             -> Bornes sont des multiples de l'amplitude testée pour qu'elles soient similaires pour tous les labels ! 
        '''
        
        if self.bins is None:  #au cas où mais inutile
            print("")
            pass
        else:            
            for rang in self.colonnes: #pour la caracteristique numero "rang"
                
                amplitude = self.bins[rang]  #son amplitude est donc ceci
                
                for label in self.labels:  #on regarde ainsi la "rang"_ième colonne de chaque label

                    premiere_valeur_serie = int(self.dictionnaire_probas[label][self.libelles[rang]][0])
                    distribution = [[(premiere_valeur_serie-premiere_valeur_serie%amplitude)-amplitude,0]]  
                    #[[x_i, n_i]]   caractere,effectif   et on fait une distribution de ses valeurs                    
                    #ce choix pour le debut de la distribtuion car si des valeurs négatives, on les prend en compte
                    #(premiere_valeur_serie-premiere_valeur_serie%amplitude) : pour que toutes les bornes soient un multiple de l'amplitude
                    #  rep - amplitude : pour que l'on commence une borne avant                
                    #intervalle :  [x; x+ amplitude[; [x+amplitude; x + 2 x amplitudes[ ; .... avec x%amplitude = 0 (multiple)
                    #choix : commencer à 0, ce n'est pas optimal en taille de mémoire, surtout pour la masse, mais c'est possible, à tester
                    
                    for valeur in self.dictionnaire_probas[label][self.libelles[rang]]:
    
                        n = 0 
                        
                        while valeur >= distribution[n][0]:                                         
                            n+=1   #on remonte 
                            if len(distribution)<n+1:                                
                                distribution.append([distribution[n-1][0]+amplitude,0]) #on rajoute une case
                          
                        distribution[n-1][1] +=1  #on est sûr que la valeur
                    
                    # -- on transforme en probabilité
                    
                    for element in distribution:
                        element[1] = element[1]/(self.dictionnaire_probas[label]["effectif"])                    
    
                    self.dictionnaire_probas[label][self.libelles[rang]] = distribution
                    
            #--- pour voir les distributions finales 
            # for element in self.dictionnaire_probas.keys():
            #     print(element,self.dictionnaire_probas[element])
            
                
    def traces_des_diverses_distributions(self):
        
        '''
        simplement pour tracer les distributions et avoir une idée des répartitions suivant les amplitudes des classes
        Utilisation de deux librairies pour tester et aussi car la seconde permet de mieux voir les distributions (quand beaucoup de classes)
        '''
        
        dossier_destination = self.reference+"_distributions/"
        creation_dossier(dossier_destination)
       
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        
        # ---- probabilités des labels 
        xi = []
        yi = []
        for label in self.dictionnaire_probas['labels'].keys():
            xi.append(label)
            yi.append(self.dictionnaire_probas['labels'][label])                
        
        ax.bar(xi,yi)
        #plt.show()
        plt.savefig(dossier_destination+'probabilites_des_labels.png', bbox_inches='tight')   
        plt.close(fig)
        
        # ---- suivant les labels, distributions des caracteristiques
        #(pas de vrais histogramme par contre mais des bar plots : manque de temps pour peaufiner)
        #plotly.graph_objects donne des fichiers html ici de plusieurs Mo, mais plus attractif que matplotlib.pyplot
                
        for label in self.labels:
            
            for caracteristique in self.dictionnaire_probas[label].keys():     
                
                if not caracteristique == "effectif":
                    xi = []
                    yi = []
                    
                    for constituant_distribution in self.dictionnaire_probas[label][caracteristique]:
                        xi.append(constituant_distribution[0])
                        yi.append(constituant_distribution[1])                    
                    
                    fig = go.Figure([go.Bar(x=xi, y=yi)])                    
            
                    fig.update_layout(
                            width=900, height=600,
                            title=label+' : distribution des '+caracteristique,
                            xaxis_title="y",
                            #yaxis_title="",
                            font=dict(family="Courier New, monospace",size=18,color='rgb(80, 80,80)'),
                            yaxis=dict(autorange=True,title="probabilité", titlefont=dict(color="#1f77b4"),tickfont=dict(color="#1f77b4"))
                            )
        
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.write_html(dossier_destination+label+'_distribution_'+caracteristique+".html")
                    del fig            
                    
                    
                    
    def reinitialisation_dictionnaire_des_probas(self):
        
        '''
        methode qui permet de remettre le dictionnaire avec les listes triees des valeurs avant determination des distributions (mise en classe) 
        cela sera utile pour la recherche des amplitudes qui optimisent les resultats du classifier quand on repartira des listes de données brutes 
        '''
        
        assert type(self.dataset) is dict, "erreur methode reinit_dict_probas"
        
        # for element in self.dataset.keys():
        #     self.dictionnaire_probas[element] = self.dataset[element]


    def actualisation(self, donnees : list):
        '''
        methode qui ajoute les données de test, une fois testées, en tant que données d'entraînement. 
        Met donc à jour les distributions en changeant les probabilités : 
            - on retourne aux effectifs dans chaque classe en multipliant les probabilités par l'ancien effectif
            - on ajoute +1 à la classe correspondante à la valeur de la nouvelle donnée : 
                > si la classe n'existe pas initialement, on la rajoute avant ou après
            - on modifie la probabilité par la formule qui évite de refaire les distribution par la méthode construction_distributions_suivant_amplitudes() : 
                p_nouvelle = (ancien_effectif +1 ) / (ancien_effectif_total +1 )
            > On stocke donc aussi la valeur de l'ancien effectif total pour le mettre à jour au fur et à mesure que l'on insère de nouvelles données.
        '''
        
        son_label = donnees[-1] #le label de la nouvelle donnée
                    
        nombre_de_donnees = 0
        for label in self.labels:
            nombre_de_donnees += self.dataset[label]['effectif'] #on calcul l'ancien effectif total
        
        # -- pour les probabilités des labels ---- 
                
        for label in self.labels:
            
            if not label == son_label:  #on change juste la probabilité du label
                self.dictionnaire_probas['labels'][label] = (nombre_de_donnees*self.dictionnaire_probas['labels'][label])/(nombre_de_donnees+1)
            else:
                #on change la probabilité du label
                self.dictionnaire_probas['labels'][label] = (nombre_de_donnees*self.dictionnaire_probas['labels'][label]+1)/(nombre_de_donnees+1)
                
                effectif = self.dictionnaire_probas[label]['effectif']

                #on change les distributions
                for rang in self.colonnes:      
                    
                    donnees_a_traiter = float(donnees[rang])             
                    
                    #si on est avant la premiere valeur de la distribution
                    if donnees_a_traiter < self.dictionnaire_probas[label][self.libelles[rang]][0][0]: 

                        borne_sup = self.dictionnaire_probas[label][self.libelles[rang]][0][0]
                        while borne_sup > donnees_a_traiter:  #on rajoute 
                            borne_sup = borne_sup-self.bins[rang]
                            self.dictionnaire_probas[label][self.libelles[rang]].insert(0,[borne_sup,0])
                        
                        self.dictionnaire_probas[label][self.libelles[rang]][0][1] = (1.)/(effectif+1)
                        self.dictionnaire_probas[label][self.libelles[rang]].insert(0,[borne_sup-self.bins[rang],0])  #on rajoute un 0 avant
                        
                        #on change les probabilités des autres
                        for u in range(2,len(self.dictionnaire_probas[label][self.libelles[rang]])):
                            self.dictionnaire_probas[label][self.libelles[rang]][u][1] = self.dictionnaire_probas[label][self.libelles[rang]][u][1]*effectif/(effectif+1)
                    
                    
                    #au contraire, si on est apres
                    elif donnees_a_traiter >= (self.dictionnaire_probas[label][self.libelles[rang]][-1][0]+self.bins[rang]): 
                        borne_inf = self.dictionnaire_probas[label][self.libelles[rang]][-1][0]
                        while borne_inf < donnees_a_traiter:  #on rajoute 
                            borne_inf = borne_inf+self.bins[rang]
                            self.dictionnaire_probas[label][self.libelles[rang]].append([borne_inf,0])
                            
                        self.dictionnaire_probas[label][self.libelles[rang]][-1][1] = 1/(effectif+1)
                        self.dictionnaire_probas[label][self.libelles[rang]].append([borne_inf+self.bins[rang],0])
                        
                        #on change les probabilités des autres
                        for u in range(len(self.dictionnaire_probas[label][self.libelles[rang]])-2):
                            self.dictionnaire_probas[label][self.libelles[rang]][u][1] = self.dictionnaire_probas[label][self.libelles[rang]][u][1]*effectif/(effectif+1)
                                        
                    else:

                        modif = False
                        for element in self.dictionnaire_probas[label][self.libelles[rang]]:
                  
                            if donnees_a_traiter<element[0]+self.bins[rang] and not modif:                                
                                element[1] = (effectif*element[1]+1)/(1+effectif)      
                                modif = True
                            else:
                                element[1] = effectif*element[1]/(1+effectif)   

                self.dictionnaire_probas[label]['effectif'] += 1  #on augmente l'effectif de 1


    def prediction_nbc(self, donnees):
        
        '''
        On va chercher à appliquer la formule 
                P(lab.|x_1,x_2) = P(x_1|lab.) \times P(x_2|lab.) \times P(lab.) 
                
        input : les données à tester
        output : le résutlat 0 Echec ou +1 Réussite
        
         '''
        
        vrai_label = donnees[-1] #le label que l'on doit trouver
        
        probabilites_suivant_le_label = [] #on stocke les valeurs des probabilités pour chaque label
        
        
        ## -------- on regarde les probabilites  P(lab.|x_1,x_2)  pour chaque label
        for label in self.labels:            
            #On initialise la probabilité par le terme P(lab.)
            proba = self.dictionnaire_probas['labels'][label]
            
            #que l'on va ensuite multiplier par chaque P(x_i |lab.)            
            for rang in self.colonnes:   
                
                valeur_a_tester = float(donnees[rang])
                distribution_a_regarder = self.dictionnaire_probas[label][self.libelles[rang]]

                if valeur_a_tester<distribution_a_regarder[0][0] or valeur_a_tester>distribution_a_regarder[-1][0]:
                    proba *= 0   #aucune valeur de référence, la probabilité est donc nulle
                else:
                    n=0
                    while valeur_a_tester > distribution_a_regarder[n][0]:
                        n+=1  #on parcourt la liste de listes
                        
                    proba *= distribution_a_regarder[n-1][1]
                    
            probabilites_suivant_le_label.append(proba) #on stocke le résultat final
        
        ## -------- on regarde quel label à la plus grande proba
        max_proba = -1
        n = -1
        doublon = False
        for probabilite_lab in probabilites_suivant_le_label:
            
            if probabilite_lab > max_proba:
                max_proba = probabilite_lab 
                n += 1
                
            elif probabilite_lab == max_proba and max_proba >-1:
                doublon = True
                n += 1
            
        
        affichage = True
        if max_proba == 0 or doublon : 
            if affichage:
                print(f"indecision par le classifier : pas de donnees ou doublon : {doublon}")
            return 0
        elif self.labels[n] == vrai_label:
            if affichage:
                print(f"pour {vrai_label}, NBC a bien prévu {self.labels[n]} ")
            return 1
        else:
            if affichage:
                print(f"pour {vrai_label}, NBC avait plutôt prévu {self.labels[n]} ")
            return 0  
        
             
        
           
        
    def apprentissage(self):
        '''               
        liste des différentes étapes/appels aux méthodes en séparant pour que cela soit plus flexible
        
        '''
        
        # 1. On regarde quels sont les labels et on trie les données en conséquence
        self.etude_des_labels_et_repartition_donnees()     

        # 2. On construit les distributions
            # a) on extrait d'abord les listes des valeurs des caractéristiques par label, que l'on va trier grâce au tri fusion, rapide, ...
        self.preconstruction_dictionnaire_probas()         
        
            # b) On a donc les series statistiques, sans distribution, 
            # on construit les distributions en fonction du choix des amplitudes/bins des intervalles
            # pour l'instant, les amplitudes sont donnees avec manchots, mais a voir ensuite si pas donnees : faire determination automatique
        
        
        if self.bins is not None:
            self.construction_distributions_suivant_amplitudes()
            
        else:
            pass ## a voir, il faut trouver une méthode qui définit les amplitudes pour chaque caractéristiques
            # il existe de tels algorithmes mais il serait intéressant de les coder soit même, comme ci-dessous : 
            # for u in range(5):  #reflechir à un critère : exemple u = (max-min)/10
            #     for v in range(6):
            #         self.bins = [u,v]
        
        
            # On peut tracer les distributions afin de voir les répartitions dans les histogrammes
        self.traces_des_diverses_distributions()
        
        
        # for element in self.dictionnaire_probas.keys():
        #     print(element,self.dictionnaire_probas[element],"\n")
                       
        
        """ 
        ## ------  Ici, on regarde maintenant si le classifier sait bien déterminer le label
        le taux de succes est actualisé et stocké au fur et à mesure dans la liste  succes_par_etapes pour voir comment s'en sort le classifier      
        
        """
        nombre_succes = 0
        nombre_elements_testes = 1
        succes_par_etapes = []
       
        for element_de_test in self.datatest:  #donnée par donnée
            resultat = self.prediction_nbc(element_de_test)  #renvoie 0 (Echec) ou 1 (Successs)
            if element_de_test[-1] != '': #si on peut améliorer les données d'entrainement
                nombre_succes += resultat
                self.actualisation(element_de_test)
                succes_par_etapes.append(round(100*nombre_succes/nombre_elements_testes,2))
                nombre_elements_testes += 1
                
        print("\n")
        print(" ______________________________________________")
        print()
        print(f"| analyse pour les données : {self.reference}")
        print(f"| {nombre_succes} succes sur {nombre_elements_testes}, soit {round(100*nombre_succes/nombre_elements_testes,2)} % de réussite |")
        print(" ______________________________________________")
        #print(nombre_elements_testes-1,len(self.datatest))
        #print( succes_par_etapes)
        plt.plot(succes_par_etapes)
        plt.ylabel('pourcentage de succes')
        plt.xlabel('etape')       
        plt.show()

        
        '''
        ********************************
        ** Améliorations envisagées : **
        ********************************
        - sauvegarde des valeurs des distributions dans des fichiers text ou csv, 
            avec lecture de ces fichiers pour les enrichir par de nouvelles données lors de nouveaux essais (on peut arrêter puis reprendre)
        - indiqué plus haut mais non encore traité : un algorithme pour trouver les amplitudes des intervalles qui maximisent les succès : 
            ici les amplitudes sont données  [5,2,10, 100] par caractéristiques, mais il faudrait trouver comment les optimiser
        '''
        
## ------------------------  Lecture d'un fichier csv, une solution parmi tant d'autres ------- 

#Ici on redonne la fonction mais ils peuvent l'avoir stockée dans leur librairie personnelle et donc
# y feront appel - point à adapter donc, suivant ce qui a été fait auparavant : modularité ! 

def csv_vers_liste_de_listes(nom_fichier_csv : str) -> list :
    
    '''
    les données ainsi extraites sont des str car il y a le nom du label, mais ce n'est pas grave, 
    car on va extraire ce label, et trnasformer les autres en nombres/float (en tout cas pour les pokemons et manchots)
    '''
    
    assert type(nom_fichier_csv) is str, "le nom du fichier n'est pas une chaîne de caractère"
    
    fichier=open(nom_fichier_csv,"r", encoding = "utf8")
    liste = [ ]
    while True:
        txt = fichier.readline()
        if txt == '':
            break
        liste.append(txt[:-1].split(","))
    fichier.close()
    return liste



## ------------------------ une simple fonction pour lister clairement les jeux de données -------

def choix_des_donnees(choix : int) -> str:
    '''    
    Une fonction rapide permettant de lister clairement les jeux de données disponibles 
    et renvoyant suivant le numéro entré dans la variable "choix", le nom du fichier correspondant.    
    
    input : un nombre entier, dont on est sûr qu'il est entier car on l'a imposé via le "int(input())" dans le main.
    output : un nom de fichier contenant les données que l'on souhaite analyser. Par défaut, celui sur les pokemons. 
             une reference sur les donnees (un nom pour identifier les fichiers)
    '''
    
    if choix == 1:
        return "liste_palmerpenguins.csv", "palmer_penguins", [5, 2, 10, 200]
    else:
        return "liste_pokemons.csv", 'pokemons', None
              

## ------------------------ ---------------------------

if __name__ == "__main__":   

    #choix = int(input("Selection du jeu de données : \n0 : pokemons  \n1 : Palmer's penguins \nvotre choix : \n "))
    #table_init = csv_vers_liste_de_listes(choix_des_donnees(choix))
    
    #-------- lecture des données 
    fichier_donnees, reference, amplitudes_proposees = choix_des_donnees(1)
    table_init = csv_vers_liste_de_listes(fichier_donnees)

    #-------- 
    libelles = table_init[0] #la dénomination exacte des libelles n'a que peu d'importance, du moment que le label est au dernier rang.
    data = table_init[1:] #slicing par simplicité et temps

    
    #-- soucis avec la liste des manchots, la liste est triée par le label (l'espèce) donc il faut remélanger : on pourrait faire autrement
    random.shuffle(data) 
    #for element in data: print(element)
    
    #-------- découpage des données en données d'entraînement et données de test  
    pourcentage_entrainement = 70
    donnees_entrainement = data[-int((pourcentage_entrainement*len(data)/100)):] #selection de 70% des données pour constituer le jeu d'entrainement
    donnees_test = data[:int((100-pourcentage_entrainement)*len(data)/100)+1] #selection des 30% restants
    #print(0.7*len(data), len(donnees_entrainement), len(donnees_test))

    #-------- creation d'une instance du classifier ------------
    A = NaiveBayesClassifier(donnees_entrainement,donnees_test,libelles,reference,amplitudes_proposees) #,[0,3])
    
    #-------- lancement de l'apprentissage et étude suivant les différents binages/amplitudes des intervalles
    A.apprentissage()
   
    #-------- application sur un manchot de type mystere  
    print("\n\n")     
    manchot_mystere = [48.4,14.6,211,4500,"Gentoo"]
    A.prediction_nbc(manchot_mystere)
    
