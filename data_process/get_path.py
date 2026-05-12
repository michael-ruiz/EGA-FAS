import os

def Get_path(data_name, prot = '1', sub_prot = None):
    # Get the project root directory (parent of data_process folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    Dir = os.path.join(project_root, 'datasets')

    train_image_dir, train_list = None, None
    val_image_dir, val_list = None, None
    test_image_dir, test_list = None, None

    if data_name == 'OULU-NPU': # no depth map
        # OULU-NPU: prot in ['1', '2', '3', '4'], sub_prot is None or ['1', '2', '3', '4', '5', '6']
        if prot == '1' or prot == '2':
            train_list = f'{Dir}/OULU-NPU-1/Prot/Protocol_{prot}/Train.txt'
            val_list = f'{Dir}/OULU-NPU-1/Prot/Protocol_{prot}/Dev.txt'
            test_list = f'{Dir}/OULU-NPU-1/Prot/Protocol_{prot}/Test.txt'
        else:
            train_list = f'{Dir}/OULU-NPU-1/Prot/Protocol_{prot}/Train_{sub_prot}.txt'
            val_list = f'{Dir}/OULU-NPU-1/Prot/Protocol_{prot}/Dev_{sub_prot}.txt'
            test_list = f'{Dir}/OULU-NPU-1/Prot/Protocol_{prot}/Test_{sub_prot}.txt'

        train_image_dir = f'{Dir}/OULU-NPU-1/'
        val_image_dir = f'{Dir}/OULU-NPU-1/'
        test_image_dir = f'{Dir}/OULU-NPU-1/'
    

    elif data_name == 'CASIA':
        train_image_dir = Dir
        train_list = f"{Dir}/cbnData/prot/CASIA_train.txt"

        val_image_dir = Dir
        val_list = f"{Dir}/cbnData/prot/CASIA_val.txt"

        test_image_dir = Dir
        test_list = f"{Dir}/cbnData/prot/CASIA_test.txt"

    elif data_name == 'RA':
        train_image_dir = Dir
        train_list = f"{Dir}/cbnData/prot/RA_train.txt"

        val_image_dir = Dir
        val_list = f"{Dir}/cbnData/prot/RA_val.txt"

        test_image_dir = Dir
        test_list = f"{Dir}/cbnData/prot/RA_test.txt"

    elif data_name == 'MSU':
        train_image_dir = f"{Dir}/MSU-MFSD"
        train_list = f"{Dir}/MSU-MFSD/train_list.txt"

        val_image_dir = f"{Dir}/MSU-MFSD"
        val_list = f"{Dir}/MSU-MFSD/test_list.txt"

        test_image_dir = f"{Dir}/MSU-MFSD"
        test_list = f"{Dir}/MSU-MFSD/test_list.txt"

    elif data_name == 'SIW':
        # SIW: prot_sub_prot in ['1_1', '2_1', '2_2', '2_3', '2_4', '3_1', '3_2']
        train_image_dir = f"{Dir}/SIW"
        train_list = f"{Dir}/SIW/train_list_{prot}_{sub_prot}.txt"

        val_image_dir = f"{Dir}/SIW"
        val_list = f"{Dir}/SIW/test_list_{prot}_{sub_prot}.txt"

        test_image_dir = f"{Dir}/SIW"
        test_list = f"{Dir}/SIW/test_list_{prot}_{sub_prot}.txt"

    elif data_name == 'CASIA-SURF':
        train_image_dir = f'{Dir}/CASIA-SURF/'
        train_list = f'{Dir}/CASIA-SURF/train_list.txt'

        val_image_dir = f'{Dir}/CASIA-SURF/'
        val_list = f'{Dir}/CASIA-SURF/val_private_list.txt'

        test_image_dir = f'{Dir}/CASIA-SURF/'
        test_list = f'{Dir}/CASIA-SURF/test_private_list.txt'

    elif data_name == 'WMCA':
        # prot1:rigidmask; prot2:replay;  prot3:prints;       prot4:papermask;
        # prot5:grandtest; prot6:glasses; prot7:flexiblemask; prot8:fakehead
        # RGB、Color、Depth、Infrared、Thermal
        NAME = 'WMCA-1'
        PROT = 'PORT2'
        train_image_dir = f'{Dir}/{NAME}/'
        train_list = f"{Dir}/{NAME}/{PROT}/{prot}_train_list.txt"

        val_image_dir = f'{Dir}/{NAME}/'
        val_list = f"{Dir}/{NAME}/{PROT}/{prot}_dev_list.txt"

        test_image_dir = f'{Dir}/{NAME}/'
        test_list = f"{Dir}/{NAME}/{PROT}/{prot}_test_list.txt"

    elif data_name == 'HQ-WMCA':
        hqwmca_dir = f'{Dir}/HQ-WMCA/'
        train_image_dir = hqwmca_dir
        train_list = f'{Dir}/HQ-WMCA/protocols/{prot}/train_list_multi.txt'
        val_image_dir = hqwmca_dir
        val_list = f'{Dir}/HQ-WMCA/protocols/{prot}/val_list_multi.txt'
        test_image_dir = hqwmca_dir
        test_list = f'{Dir}/HQ-WMCA/protocols/{prot}/test_list_multi.txt'

    elif data_name == 'CASIA-FASD':
        casia_fasd_dir = f'{Dir}/CASIA-FASD/'
        train_image_dir = casia_fasd_dir
        train_list = f'{Dir}/CASIA-FASD/protocol/train_list.txt'
        val_image_dir = casia_fasd_dir
        val_list = f'{Dir}/CASIA-FASD/protocol/val_list.txt'
        test_image_dir = casia_fasd_dir
        test_list = f'{Dir}/CASIA-FASD/protocol/test_list.txt'

    elif data_name == 'Replay-Attack':
        ra_dir = f'{Dir}/Replay-Attack/'
        train_image_dir = ra_dir
        train_list = f'{Dir}/Replay-Attack/protocol/train_list.txt'
        val_image_dir = ra_dir
        val_list = f'{Dir}/Replay-Attack/protocol/val_list.txt'
        test_image_dir = ra_dir
        test_list = f'{Dir}/Replay-Attack/protocol/test_list.txt'

    elif data_name == 'VFPAD':
        # Single NIR modality; prot must be 'grandtest'
        vfpad_dir = f'{Dir}/VFPAD/'
        train_image_dir = vfpad_dir
        train_list = f'{Dir}/VFPAD/protocol/grandtest/train_list.txt'

        val_image_dir = vfpad_dir
        val_list = f'{Dir}/VFPAD/protocol/grandtest/dev_list.txt'

        test_image_dir = vfpad_dir
        test_list = f'{Dir}/VFPAD/protocol/grandtest/eval_list.txt'

    # 跨数据及测试CASIA-RA
    elif data_name == 'CASIA-RA':
        train_image_dir = Dir
        train_list = f"{Dir}/cbnData/prot/CASIA_train.txt"

        val_image_dir = Dir
        val_list = f"{Dir}/cbnData/prot/CASIA_val.txt"

        test_image_dir = Dir
        test_list =f"{Dir}/cbnData/prot/RA_test.txt"

    elif data_name == 'RA-CASIA':
        train_image_dir = Dir
        train_list = f"{Dir}/cbnData/prot/RA_train.txt"

        val_image_dir = Dir
        val_list = f"{Dir}/cbnData/prot/RA_val.txt"

        test_image_dir = Dir
        test_list = f"{Dir}/cbnData/prot/CASIA_test.txt"

    # Cross-dataset: WMCA ↔ HQ-WMCA (single-modal visible only)
    # WMCA color = grayscale visible; HQ-WMCA visible = Ch0 grayscale
    # Must use --num_modalities=1 (single-modal) since modalities differ
    elif data_name == 'WMCA-HQWMCA':
        NAME = 'WMCA-1'
        PROT = 'PORT2'
        train_image_dir = f'{Dir}/{NAME}/'
        train_list = f"{Dir}/{NAME}/{PROT}/prot5_train_list_color.txt"
        val_image_dir = f'{Dir}/{NAME}/'
        val_list = f"{Dir}/{NAME}/{PROT}/prot5_dev_list_color.txt"
        test_image_dir = f'{Dir}/HQ-WMCA/'
        test_list = f'{Dir}/HQ-WMCA/protocols/grand_test-curated/test_list_visible.txt'

    elif data_name == 'HQWMCA-WMCA':
        hqwmca_dir = f'{Dir}/HQ-WMCA/'
        train_image_dir = hqwmca_dir
        train_list = f'{Dir}/HQ-WMCA/protocols/grand_test-curated/train_list_visible.txt'
        val_image_dir = hqwmca_dir
        val_list = f'{Dir}/HQ-WMCA/protocols/grand_test-curated/val_list_visible.txt'
        NAME = 'WMCA-1'
        PROT = 'PORT2'
        test_image_dir = f'{Dir}/{NAME}/'
        test_list = f"{Dir}/{NAME}/{PROT}/prot5_test_list_color.txt"

    # Cross-dataset: CASIA-FASD ↔ Replay-Attack (single-modal)
    # Uses MTCNN-cropped frames for both datasets (consistent face detection)
    elif data_name == 'CASIA_FASD-RA':
        casia_fasd_dir = f'{Dir}/CASIA-FASD/'
        train_image_dir = casia_fasd_dir
        train_list = f'{Dir}/CASIA-FASD/protocol/train_list.txt'
        val_image_dir = casia_fasd_dir
        val_list = f'{Dir}/CASIA-FASD/protocol/val_list.txt'
        ra_dir = f'{Dir}/Replay-Attack/'
        test_image_dir = ra_dir
        test_list = f'{Dir}/Replay-Attack/protocol/test_list_mtcnn.txt'

    elif data_name == 'RA-CASIA_FASD':
        ra_dir = f'{Dir}/Replay-Attack/'
        train_image_dir = ra_dir
        train_list = f'{Dir}/Replay-Attack/protocol/train_list_mtcnn.txt'
        val_image_dir = ra_dir
        val_list = f'{Dir}/Replay-Attack/protocol/val_list_mtcnn.txt'
        casia_fasd_dir = f'{Dir}/CASIA-FASD/'
        test_image_dir = casia_fasd_dir
        test_list = f'{Dir}/CASIA-FASD/protocol/test_list.txt'

    train_path = {'image_dir': train_image_dir, 'prot_list': train_list}
    val_path = {'image_dir': val_image_dir, 'prot_list': val_list}
    test_path = {'image_dir': test_image_dir, 'prot_list': test_list}

    return train_path, val_path, test_path