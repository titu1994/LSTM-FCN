
TRAIN_FILES = ['../data\\Adiac_TRAIN', # 0
               '../data\\ArrowHead_TRAIN',  # 1
               '../data\\ChlorineConcentration_TRAIN',  # 2
               '../data\\InsectWingbeatSound_TRAIN',  # 3
               '../data\\Lighting7_TRAIN',  # 4
               '../data\\Wine_TRAIN',  # 5
               '../data\\WordsSynonyms_TRAIN',  # 6
               '../data\\50words_TRAIN',  # 7
               '../data\\Beef_TRAIN',  # 8
               '../data\\DistalPhalanxOutlineAgeGroup_TEST',  # 9 (inverted dataset)
               '../data\\DistalPhalanxOutlineCorrect_TEST',  # 10 (inverted dataset)
               '../data\\DistalPhalanxTW_TEST',  # 11 (inverted dataset)
               '../data\\ECG200_TRAIN',  # 12
               '../data\\ECGFiveDays_TRAIN',  # 13
               '../data\\BeetleFly_TRAIN',  # 14
               '../data\\BirdChicken_TRAIN',  # 15
               '../data\\ItalyPowerDemand_TRAIN', # 16
               '../data\\SonyAIBORobotSurface_TRAIN', # 17
               '../data\\SonyAIBORobotSurfaceII_TRAIN', # 18
               '../data\\MiddlePhalanxOutlineAgeGroup_TEST', # 19
               '../data\\MiddlePhalanxOutlineCorrect_TEST', # 20
               '../data\\MiddlePhalanxTW_TEST', # 21
               '../data\\ProximalPhalanxOutlineAgeGroup_TRAIN', # 22
               '../data\\ProximalPhalanxOutlineCorrect_TRAIN', # 23
               '../data\\ProximalPhalanxTW_TEST', # 24 (inverted dataset)
               '../data\\MoteStrain_TRAIN', # 25
               '../data\\MedicalImages_TRAIN', # 26
               '../data\\Strawberry_TEST',  # 27 (inverted dataset)
               '../data\\ToeSegmentation1_TRAIN',  # 28
               '../data\\Coffee_TRAIN',  # 29
               '../data\\Cricket_X_TRAIN',  # 30
               '../data\\Cricket_Y_TRAIN',  # 31
               '../data\\Cricket_Z_TRAIN',  # 32
               '../data\\uWaveGestureLibrary_X_TRAIN',  # 33
               '../data\\uWaveGestureLibrary_Y_TRAIN',  # 34
               '../data\\uWaveGestureLibrary_Z_TRAIN',  # 35
               '../data\\ToeSegmentation2_TRAIN',  # 36
               '../data\\DiatomSizeReduction_TRAIN',  # 37
               ]

TEST_FILES = ['../data\\Adiac_TEST', # 0
              '../data\\ArrowHead_TEST', # 1
              '../data\\ChlorineConcentration_TEST', # 2
              '../data\\InsectWingbeatSound_TEST', # 3
              '../data\\Lighting7_TEST', # 4
              '../data\\Wine_TEST', # 5
              '../data\\WordsSynonyms_TEST', # 6
              '../data\\50words_TEST', # 7
              '../data\\Beef_TEST', # 8
              '../data\\DistalPhalanxOutlineAgeGroup_TRAIN', # 9 (inverted dataset)
              '../data\\DistalPhalanxOutlineCorrect_TRAIN', # 10 (inverted dataset)
              '../data\\DistalPhalanxTW_TRAIN', # 11 (inverted dataset)
              '../data\\ECG200_TEST', # 12
              '../data\\ECGFiveDays_TEST', # 13
              '../data\\BeetleFly_TEST', # 14
              '../data\\BirdChicken_TEST', # 15
              '../data\\ItalyPowerDemand_TEST', # 16
              '../data\\SonyAIBORobotSurface_TEST', # 17
              '../data\\SonyAIBORobotSurfaceII_TEST', # 18
              '../data\\MiddlePhalanxOutlineAgeGroup_TRAIN', # 19 (inverted dataset)
              '../data\\MiddlePhalanxOutlineCorrect_TRAIN', # 20 (inverted dataset)
              '../data\\MiddlePhalanxTW_TRAIN', # 21 (inverted dataset)
              '../data\\ProximalPhalanxOutlineAgeGroup_TEST', # 22
              '../data\\ProximalPhalanxOutlineCorrect_TEST', # 23
              '../data\\ProximalPhalanxTW_TRAIN', # 24 (inverted dataset)
              '../data\\MoteStrain_TEST', # 25
              '../data\\MedicalImages_TEST', # 26
              '../data\\Strawberry_TRAIN',  # 27
              '../data\\ToeSegmentation1_TEST',  # 28
              '../data\\Coffee_TEST',  # 29
              '../data\\Cricket_X_TEST',  # 30
              '../data\\Cricket_Y_TEST',  # 31
              '../data\\Cricket_Z_TEST',  # 32
              '../data\\uWaveGestureLibrary_X_TEST',  # 33
              '../data\\uWaveGestureLibrary_Y_TEST',  # 34
              '../data\\uWaveGestureLibrary_Z_TEST',  # 35
              '../data\\ToeSegmentation2_TEST',  # 36
              '../data\\DiatomSizeReduction_TEST',  # 37
              ]

# Not used anymore
MAX_NB_WORDS_LIST = [17, # 0
                     17, # 1
                     17, # 2
                     17, # 3
                     17, # 4
                     257, # 5
                     17, # 6
                     4097, # 7
                     4097, # 8
                     17, # 9
                     17, # 10
                     17, # 11
                     17, # 12
                     4097, # 13
                     4097, # 14
                     4097, # 15
                     17, # 16
                     17, # 17
                     17, # 18
                     17, # 19
                     17, # 20
                     17, # 21
                     17, # 22
                     17, # 23
                     17, # 24
                     17, # 25
                     17, # 26
                     17,  # 27
                     17,  # 28
                     17,  # 29
                     17,  # 30
                     17,  # 31
                     17,  # 32
                     17,  # 33
                     17,  # 34
                     17,  # 35
                     17,  # 36
                     17,  # 37
                     ]

MAX_SEQUENCE_LENGTH_LIST = [176, # 0
                            251, # 1
                            166, # 2
                            256, # 3
                            257, # 4
                            234, # 5
                            270, # 6
                            270, # 7
                            470, # 8
                            80,  # 9
                            80,  # 10
                            80,  # 11
                            96, # 12
                            136, # 13
                            512, # 14
                            512, # 15
                            24, # 16
                            70, # 17
                            65, # 18
                            80, # 19
                            80, # 20
                            80, # 21
                            80, # 22
                            80, # 23
                            80, # 24
                            84, # 25
                            99, # 26
                            235, # 27
                            277, # 28
                            286, # 29
                            300, # 30
                            300, # 31
                            300, # 32
                            315, # 33
                            315, # 34
                            315, # 35
                            343, # 36
                            345, # 37
                            ]

NB_CLASSES_LIST = [37, # 0
                   3, # 1
                   3, # 2
                   11, # 3
                   7, # 4
                   2, # 5
                   25, # 6
                   50, # 7
                   5, # 8
                   3, # 9
                   2, # 10
                   6, # 11
                   2, # 12
                   2, # 13
                   2, # 14
                   2, # 15
                   2, # 16
                   2, # 17
                   2, # 18
                   3, # 19
                   2, # 20
                   6, # 21
                   3, # 22
                   2, # 23
                   6, # 24
                   2, # 25
                   10, # 26
                   2,  # 27
                   2,  # 28
                   2,  # 29
                   12,  # 30
                   12,  # 31
                   12,  # 32
                   8,  # 33
                   8,  # 34
                   8,  # 35
                   2,  # 36
                   4,  # 37
                   ]