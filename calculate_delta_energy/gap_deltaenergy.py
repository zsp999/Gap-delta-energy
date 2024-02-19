from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd
import networkx as nx
from rdkit.Chem.rdchem import BondType

def GetNBEandGapEnergy(mols, smiles_list, res_error='results/all_gapenergy_error.txt',):
    res_error = open(res_error,'a')
    descs = [desc_name[0] for desc_name in Descriptors._descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    atoms = [['H', 'H'], ['H', 'F'], ['H', 'Cl'], ['H', 'Br'], ['H', 'I'],
             ['C', 'H'], ['C', 'C'], ['C', 'N'], ['C', 'O'], ['C', 'F'], ['C', 'Cl'], ['C', 'Br'], ['C', 'I'], ['C', 'S'],
             ['N', 'H'], ['N', 'N'], ['N', 'F'], ['N', 'Cl'], ['N', 'Br'], ['N', 'O'],
             ['O', 'H'], ['O', 'O'], ['O', 'F'], ['O', 'Cl'], ['O', 'I'],
             ['F', 'F'], ['F', 'Cl'], ['F', 'Br'], ['Cl', 'Cl'], ['Cl', 'Br'], ['Br', 'Br'], ['I', 'I'], ['I', 'Cl'], ['I', 'Br'],
             ['S', 'H'], ['S', 'F'], ['S', 'Cl'], ['S', 'Br'], ['S', 'S'],
             ['Si', 'Si'], ['Si', 'H'], ['Si', 'C'], ['Si', 'O'],
             ['C', 'C'], ['C', 'C'], ['O', 'O'], ['C', 'O'], ['C', 'O'], ['N', 'O'], ['N', 'N'],
             ['N', 'N'], ['C', 'N'], ['C', 'N'],
             ['P', 'H'], ['P', 'Cl'], ['P', 'Br'], ['P', 'O'], ['P', 'O'], ['P', 'P'], ['P', 'C'],
             ['S', 'C'], ['S', 'O'], ['S', 'O'],
             ['S','N'], ['S','P'], ['S','P']]
    bondtypes = [BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, # 5
                 BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, #9
                 BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, #6
                 BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, #5
                 BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, #9
                 BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, #5
                 BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, #4
                 BondType.DOUBLE, BondType.TRIPLE, BondType.DOUBLE, BondType.DOUBLE, BondType.TRIPLE, BondType.DOUBLE, BondType.DOUBLE, #7
                 BondType.TRIPLE, BondType.TRIPLE, BondType.DOUBLE, #3
                 BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.DOUBLE, BondType.SINGLE, BondType.SINGLE, #7
                 BondType.DOUBLE, BondType.SINGLE, BondType.DOUBLE, #3
                 BondType.SINGLE, BondType.SINGLE, BondType.DOUBLE #3
                 ]
    energies = [432, 565, 427, 363, 295, 413, 347, 305, 358, 485, 339, 276, 240, 259, 391, 160, 272, 200, 243, 201, 467,
                146, 190, 203, 234,
                154, 253, 237, 239, 218, 193, 149, 208, 175, 347, 327, 253, 218, 266, 340, 393, 360, 452, 614, 839, 495,
                745, 1072, 607, 418,
                941, 891, 615, 322, 331, 272, 410, 585, 213, 305, 536, 226, 460,
                121, 463, 389]
    # 键长参考https://wenku.baidu.com/view/d9284346767f5acfa1c7cd42.html#opennewwindow
    # http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
    # https://cccbdb.nist.gov/calcbondcomp2.asp?i=7&j=35
    ExactbondWet = [2.015650064, 20.006228252, 35.976677712, 79.926162132, 127.912298032, 13.007825032, 24.0,
                    26.003073999999998, 27.99491462, 30.99840322, 46.96885268,
                    90.9183371, 138.904473, 43.972071, 15.010899032, 28.006148, 33.00147722, 48.971926679999996,
                    92.9214111, 29.99798862, 17.002739652, 31.98982924,
                    34.99331784, 50.9637673, 142.89938762, 37.99680644, 53.9672559, 97.91674032, 69.93770536,
                    113.88718978, 157.8366742, 253.808946,
                    161.87332568, 205.8228101, 32.979896032, 50.97047422, 66.94092368, 110.8904081, 63.944142,
                    55.95385306, 28.984751562, 39.97692653,
                    43.97184115, 24.0, 24.0, 31.98982924, 27.99491462, 27.99491462, 29.99798862, 28.006148, 28.006148,
                    26.003073999999998, 26.003073999999998,
                    31.981586661999998, 65.94261431, 109.89209873, 46.96867625, 46.96867625, 61.94752326, 42.97376163, \
                    43.972071, 47.96698562, 47.96698562,
                    45.975145, 62.94583263, 62.94583263]

    icount = 0
    suscount = 0
    errcount = 0
    nbegapenergy = []
    while True:

        if icount>= len(smiles_list):
            break
        mol = mols[icount]

        try:
            bindnum = len(mol.GetBonds())
            if bindnum > 2:
                pass
            else:
                res_error.write(str(icount) + '\t' + 'num_bond < 3.' + '\t' + str(smiles_list[icount]) + '\n')
                errcount += 1
                del mol[icount]
                del smiles_list[icount]

                continue
        except:
            res_error.write(str(icount) + '\t' + 'cannot get bond.' + '\t' + str(smiles_list[icount]) + '\n')
            errcount += 1
            del mols[icount]
            del smiles_list[icount]
            continue

        try:
            mol1 = Chem.AddHs(mol)  # RDkit中默认不显示氢,向分子中添加H
        except:
            res_error.write(str(icount) + '\t' + 'AddHs error.' + '\t' + str(smiles_list[icount]) + '\n')
            errcount += 1
            del mols[icount]
            del smiles_list[icount]
            continue
        try:
            Chem.Kekulize(mol1)  # 向分子中添加芳香共轭键
        except:
            res_error.write(str(icount) + '\t' + 'Addkekus error.' + '\t' + str(smiles_list[icount]) + '\n')
            errcount += 1
            del mols[icount]
            del smiles_list[icount]
            continue

        flag = False
        # newbondenergy = []
        # oldbondenergy = []
        list_of_numatompair = []
        for bond in mol1.GetBonds():  # 读取mol中所有化学键，得到所有键的列表，遍历
            flag = False
            bondenergy = 0.0
            beginnum = mol1.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetIdx()
            endnum = mol1.GetAtomWithIdx(bond.GetEndAtomIdx()).GetIdx()
            atompair = [mol1.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol(),mol1.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()]

            for i in range(len(atoms)):  # 一个个试，atoms是数据库里的已知所有的原子对
                if (sorted(atompair) == sorted(atoms[i])) and (bond.GetBondType() == bondtypes[i]):  # 一个个比对
                    flag = True
                    bondenergy = energies[i]
                    # newbondenergy.append(bondenergy / ExactbondWet[i])
                    # oldbondenergy.append(bondenergy)
            if flag == True:
                list_of_numatompair.append([beginnum, endnum, bondenergy])
                pass
            else:
                res_error.write(
                    str(icount) + '\t' + f'cannot calculate bond energy {atompair[0]} and {atompair[1]},{bond.GetBondType()}.' + '\t'  +
                    str(smiles_list[icount]) + '\n')
                errcount += 1
                break
        if flag == False:
            del mols[icount]
            del smiles_list[icount]
            continue
        else:
            pass

        # newstd = np.std(newbondenergy)
        # newnbe = sum(newbondenergy) / len(mol1.GetBonds())
        # oldstd = np.std(oldbondenergy)
        # oldnbe = sum(oldbondenergy) / Descriptors.ExactMolWt(mol1)

        G = nx.Graph()
        num = len(list_of_numatompair)

        nodelist = [[i, list_of_numatompair[i][2]] for i in range(num)]

        edgelist = []
        for i in range(num):
            a1 = list_of_numatompair[i][0]
            b1 = list_of_numatompair[i][1]
            for x in range(num):
                if list_of_numatompair[x][0] == a1 or list_of_numatompair[x][1] == a1:
                    edgelist.append((i, x))
                if list_of_numatompair[x][0] == b1 or list_of_numatompair[x][1] == b1:
                    edgelist.append((i, x))
        edgelistdel = []
        for edge in edgelist:
            if edge[0] == edge[1]:
                edgelistdel.append(edge)
        for edgedel in edgelistdel:
            edgelist.remove(edgedel)


        G.add_nodes_from([i[0] for i in nodelist])
        G.add_edges_from(edgelist)

        pathlist = []
        flag = True

        length_source_target = dict(nx.shortest_path_length(G))
        gap_and_energy = []
        for idx1 in range(num):
            for idx2 in range(idx1+1,num):
                try:
                    gap = length_source_target[idx1][idx2]
                    startenergy = nodelist[idx1][1]
                    endenergy = nodelist[idx2][1]
                    gapenergy = abs(startenergy-endenergy)
                    gap_and_energy.append([gap, gapenergy])
                except:
                    res_error.write(str(icount) + '\t' + smiles_list[icount]+ '\t' +f'exist node without path {idx1} and {idx2}'+'\n')
                    flag = False
                    break
            if flag == False:
                break

        if flag == False:
            errcount += 1
            del mols[icount]
            del smiles_list[icount]
            continue

    
        max_num = 1 # max_num是一个分子的最大gap
        for i in gap_and_energy:
            max_num = max(max_num, i[0])
        # print(max_num)
        # arrlist是储存各个gap下平均键能差的数组的列表
        arrdata = []
        for i in range(max_num):  # range(max_num)
            arrlist = []
            for x in gap_and_energy:
                if x[0] == i + 1:
                    arrlist.append(x[1])
            arrdata.append(arrlist)

        dimension = 65

        nbegapenergy_mol=[]
        #sxmean, sxstd, nxmean, nxstd
        for i in range(dimension):
            try:
                nbegapenergy_mol.append(np.mean(arrdata[i])) #np.mean(
            except:
                nbegapenergy_mol.append(np.nan) #[]

        # nbegapenergy_mol.extend([oldnbe, oldstd, newnbe, newstd, max_num])
        nbegapenergy.append(nbegapenergy_mol)
        icount += 1
        suscount += 1
    return (nbegapenergy, smiles_list)

def GetNBEandGapEnergy_asfeatures(mols, smiles_list, res_error='tox_gapenergy_error.txt',):
    res_error = open(res_error,'a')
    descs = [desc_name[0] for desc_name in Descriptors._descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    atoms = [['H', 'H'], ['H', 'F'], ['H', 'Cl'], ['H', 'Br'], ['H', 'I'],
             ['C', 'H'], ['C', 'C'], ['C', 'N'], ['C', 'O'], ['C', 'F'], ['C', 'Cl'], ['C', 'Br'], ['C', 'I'], ['C', 'S'],
             ['N', 'H'], ['N', 'N'], ['N', 'F'], ['N', 'Cl'], ['N', 'Br'], ['N', 'O'],
             ['O', 'H'], ['O', 'O'], ['O', 'F'], ['O', 'Cl'], ['O', 'I'],
             ['F', 'F'], ['F', 'Cl'], ['F', 'Br'], ['Cl', 'Cl'], ['Cl', 'Br'], ['Br', 'Br'], ['I', 'I'], ['I', 'Cl'], ['I', 'Br'],
             ['S', 'H'], ['S', 'F'], ['S', 'Cl'], ['S', 'Br'], ['S', 'S'],
             ['Si', 'Si'], ['Si', 'H'], ['Si', 'C'], ['Si', 'O'],
             ['C', 'C'], ['C', 'C'], ['O', 'O'], ['C', 'O'], ['C', 'O'], ['N', 'O'], ['N', 'N'],
             ['N', 'N'], ['C', 'N'], ['C', 'N'],
             ['P', 'H'], ['P', 'Cl'], ['P', 'Br'], ['P', 'O'], ['P', 'O'], ['P', 'P'], ['P', 'C'],
             ['S', 'C'], ['S', 'O'], ['S', 'O'],
             ['S','N'], ['S','P'], ['S','P']]
    bondtypes = [BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, # 5
                 BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, #9
                 BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, #6
                 BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, #5
                 BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, #9
                 BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, #5
                 BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, #4
                 BondType.DOUBLE, BondType.TRIPLE, BondType.DOUBLE, BondType.DOUBLE, BondType.TRIPLE, BondType.DOUBLE, BondType.DOUBLE, #7
                 BondType.TRIPLE, BondType.TRIPLE, BondType.DOUBLE, #3
                 BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.SINGLE, BondType.DOUBLE, BondType.SINGLE, BondType.SINGLE, #7
                 BondType.DOUBLE, BondType.SINGLE, BondType.DOUBLE, #3
                 BondType.SINGLE, BondType.SINGLE, BondType.DOUBLE #3
                 ]
    energies = [432, 565, 427, 363, 295, 413, 347, 305, 358, 485, 339, 276, 240, 259, 391, 160, 272, 200, 243, 201, 467,
                146, 190, 203, 234,
                154, 253, 237, 239, 218, 193, 149, 208, 175, 347, 327, 253, 218, 266, 340, 393, 360, 452, 614, 839, 495,
                745, 1072, 607, 418,
                941, 891, 615, 322, 331, 272, 410, 585, 213, 305, 536, 226, 460,
                121, 463, 389]
    # 键长参考https://wenku.baidu.com/view/d9284346767f5acfa1c7cd42.html#opennewwindow
    # http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
    # https://cccbdb.nist.gov/calcbondcomp2.asp?i=7&j=35
    ExactbondWet = [2.015650064, 20.006228252, 35.976677712, 79.926162132, 127.912298032, 13.007825032, 24.0,
                    26.003073999999998, 27.99491462, 30.99840322, 46.96885268,
                    90.9183371, 138.904473, 43.972071, 15.010899032, 28.006148, 33.00147722, 48.971926679999996,
                    92.9214111, 29.99798862, 17.002739652, 31.98982924,
                    34.99331784, 50.9637673, 142.89938762, 37.99680644, 53.9672559, 97.91674032, 69.93770536,
                    113.88718978, 157.8366742, 253.808946,
                    161.87332568, 205.8228101, 32.979896032, 50.97047422, 66.94092368, 110.8904081, 63.944142,
                    55.95385306, 28.984751562, 39.97692653,
                    43.97184115, 24.0, 24.0, 31.98982924, 27.99491462, 27.99491462, 29.99798862, 28.006148, 28.006148,
                    26.003073999999998, 26.003073999999998,
                    31.981586661999998, 65.94261431, 109.89209873, 46.96867625, 46.96867625, 61.94752326, 42.97376163, \
                    43.972071, 47.96698562, 47.96698562,
                    45.975145, 62.94583263, 62.94583263]

    icount = 0
    suscount = 0
    errcount = 0
    nbegapenergy = []
    while True:

        if icount>= len(smiles_list):
            break
        mol = mols[icount]

        try:
            bindnum = len(mol.GetBonds())
            if bindnum > 2:
                pass
            else:
                res_error.write(str(icount) + '\t' + 'num_bond < 3.' + '\t' + str(smiles_list[icount]) + '\n')
                errcount += 1
                del mol[icount]
                del smiles_list[icount]

                continue
        except:
            res_error.write(str(icount) + '\t' + 'cannot get bond.' + '\t' + str(smiles_list[icount]) + '\n')
            errcount += 1
            del mols[icount]
            del smiles_list[icount]
            continue

        try:
            mol1 = Chem.AddHs(mol)  # RDkit中默认不显示氢,向分子中添加H
        except:
            res_error.write(str(icount) + '\t' + 'AddHs error.' + '\t' + str(smiles_list[icount]) + '\n')
            errcount += 1
            del mols[icount]
            del smiles_list[icount]
            continue
        try:
            Chem.Kekulize(mol1)  # 向分子中添加芳香共轭键
        except:
            res_error.write(str(icount) + '\t' + 'Addkekus error.' + '\t' + str(smiles_list[icount]) + '\n')
            errcount += 1
            del mols[icount]
            del smiles_list[icount]
            continue

        flag = False
        newbondenergy = []
        oldbondenergy = []
        list_of_numatompair = []
        for bond in mol1.GetBonds():  # 读取mol中所有化学键，得到所有键的列表，遍历
            flag = False
            bondenergy = 0.0
            beginnum = mol1.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetIdx()
            endnum = mol1.GetAtomWithIdx(bond.GetEndAtomIdx()).GetIdx()
            atompair = [mol1.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol(),mol1.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()]

            for i in range(len(atoms)):  # 一个个试，atoms是数据库里的已知所有的原子对
                if (sorted(atompair) == sorted(atoms[i])) and (bond.GetBondType() == bondtypes[i]):  # 一个个比对
                    flag = True
                    bondenergy = energies[i]
                    newbondenergy.append(bondenergy / ExactbondWet[i])
                    oldbondenergy.append(bondenergy)
            if flag == True:
                list_of_numatompair.append([beginnum, endnum, bondenergy])
                pass
            else:
                res_error.write(
                    str(icount) + '\t' + f'cannot calculate bond energy {atompair[0]} and {atompair[1]},{bond.GetBondType()}.' + '\t'  +
                    str(smiles_list[icount]) + '\n')
                errcount += 1
                break
        if flag == False:
            del mols[icount]
            del smiles_list[icount]
            continue
        else:
            pass

        newstd = np.std(newbondenergy)
        newnbe = sum(newbondenergy) / len(mol1.GetBonds())
        oldstd = np.std(oldbondenergy)
        oldnbe = sum(oldbondenergy) / Descriptors.ExactMolWt(mol1)

        G = nx.Graph()
        num = len(list_of_numatompair)

        nodelist = [[i, list_of_numatompair[i][2]] for i in range(num)]

        edgelist = []
        for i in range(num):
            a1 = list_of_numatompair[i][0]
            b1 = list_of_numatompair[i][1]
            for x in range(num):
                if list_of_numatompair[x][0] == a1 or list_of_numatompair[x][1] == a1:
                    edgelist.append((i, x))
                if list_of_numatompair[x][0] == b1 or list_of_numatompair[x][1] == b1:
                    edgelist.append((i, x))
        edgelistdel = []
        for edge in edgelist:
            if edge[0] == edge[1]:
                edgelistdel.append(edge)
        for edgedel in edgelistdel:
            edgelist.remove(edgedel)


        G.add_nodes_from([i[0] for i in nodelist])
        G.add_edges_from(edgelist)

        pathlist = []
        flag = True

        length_source_target = dict(nx.shortest_path_length(G))
        gap_and_energy = []
        for idx1 in range(num):
            for idx2 in range(idx1+1,num):
                try:
                    gap = length_source_target[idx1][idx2]
                    startenergy = nodelist[idx1][1]
                    endenergy = nodelist[idx2][1]
                    gapenergy = abs(startenergy-endenergy)
                    gap_and_energy.append([gap, gapenergy])
                except:
                    res_error.write(str(icount) + '\t' + smiles_list[icount]+ '\t' +f'exist node without path {idx1} and {idx2}'+'\n')
                    flag = False
                    break
            if flag == False:
                break

        if flag == False:
            errcount += 1
            del mols[icount]
            del smiles_list[icount]
            continue

    
        max_num = 1 # max_num是一个分子的最大gap
        for i in gap_and_energy:
            max_num = max(max_num, i[0])
        # print(max_num)
        # arrlist是储存各个gap下平均键能差的数组的列表
        arrdata = []
        for i in range(max_num):  # range(max_num)
            arrlist = []
            for x in gap_and_energy:
                if x[0] == i + 1:
                    arrlist.append(x[1])
            arrdata.append(arrlist)

        dimension = 10

        nbegapenergy_mol=[]
        #sxmean, sxstd, nxmean, nxstd
        for i in range(dimension):
            try:
                nbegapenergy_mol.append(np.mean(arrdata[i]))
            except:
                nbegapenergy_mol.append(np.nan)
        for i in range(dimension):
            try:
                nbegapenergy_mol.append(np.std(arrdata[i]))
            except:
                nbegapenergy_mol.append(np.nan) 
        for i in range(dimension):
            try:
                nbegapenergy_mol.append(np.mean(arrdata[-i-1]))
            except:
                nbegapenergy_mol.append(np.nan) 
        for i in range(dimension):
            try:
                nbegapenergy_mol.append(np.std(arrdata[-i-1]))
            except:
                nbegapenergy_mol.append(np.nan) 
        nbegapenergy_mol.extend([oldnbe, oldstd, newnbe, newstd, max_num])
        nbegapenergy.append(nbegapenergy_mol)
        icount += 1
        suscount += 1
    return nbegapenergy, smiles_list


col_list = ['sx_ave1', 'sx_ave2','sx_ave3','sx_ave4','sx_ave5',
            'sx_ave6',  'sx_ave7','sx_ave8','sx_ave9','sx_ave10',
            'sx_std1', 'sx_std2', 'sx_std3', 'sx_std4', 'sx_std5', 
            'sx_std6', 'sx_std7', 'sx_std8', 'sx_std9', 'sx_std10', 
            'nx_ave1', 'nx_ave2','nx_ave3','nx_ave4','nx_ave5',
            'nx_ave6',  'nx_ave7','nx_ave8','nx_ave9','nx_ave10',
            'nx_std1', 'nx_std2', 'nx_std3', 'nx_std4', 'nx_std5', 
            'nx_std6', 'nx_std7', 'nx_std8', 'nx_std9', 'nx_std10', 
            'oldnbe', 'oldstd', 'newnbe', 'newstd', 'max_gap']

# feature engineer
def FE(df_gapenergy):
    # df_gapenergy_fe = df_gapenergy.copy()
    df_gapenergy_fe = []
    df_colnames = []
    for idx1 in range(10):
        for idx2 in range(idx1+1, 10):
            str_col = str(col_list[idx1])+'*'+str(col_list[idx2])
            # df_gapenergy_fe[str_col] = df_gapenergy[col_list[idx1]]*df_gapenergy[col_list[idx2]]
            df_gapenergy_fe.append(df_gapenergy[col_list[idx1]]*df_gapenergy[col_list[idx2]])
            df_colnames.append(str_col)
    for idx1 in range(10, 20):
        for idx2 in range(idx1+1, 20):
            str_col = str(col_list[idx1])+'*'+str(col_list[idx2])
            # df_gapenergy_fe[str_col] = df_gapenergy[col_list[idx1]]*df_gapenergy[col_list[idx2]]
            df_gapenergy_fe.append(df_gapenergy[col_list[idx1]]*df_gapenergy[col_list[idx2]])
            df_colnames.append(str_col)            
    for idx1 in range(20,30):
        for idx2 in range(idx1+1, 30):
            str_col = str(col_list[idx1])+'*'+str(col_list[idx2])
            # df_gapenergy_fe[str_col] = df_gapenergy[col_list[idx1]]*df_gapenergy[col_list[idx2]]
            df_gapenergy_fe.append(df_gapenergy[col_list[idx1]]*df_gapenergy[col_list[idx2]])
            df_colnames.append(str_col)
    for idx1 in range(30,40):
        for idx2 in range(idx1+1, 40):
            str_col = str(col_list[idx1])+'*'+str(col_list[idx2])
            # df_gapenergy_fe[str_col] = df_gapenergy[col_list[idx1]]*df_gapenergy[col_list[idx2]]
            df_gapenergy_fe.append(df_gapenergy[col_list[idx1]]*df_gapenergy[col_list[idx2]])
            df_colnames.append(str_col)
    
    df_gapenergy_fe =  pd.concat(df_gapenergy_fe, axis=1)
    df_gapenergy_fe.columns = df_colnames
    df_gapenergy_fe = pd.concat([df_gapenergy, df_gapenergy_fe], axis=1)
    
    return df_gapenergy_fe

# molecular descriptors
descList = [i[0] for i in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descList)

def descri(df: pd.DataFrame):
    descriptors = []
    smis = df['standard_smiles'].values
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        descriptors.append(list(calculator.CalcDescriptors(mol)))
    df_des = pd.DataFrame(descriptors, columns=descList )
    df_all = pd.concat([df, df_des], axis=1)
    return df_all