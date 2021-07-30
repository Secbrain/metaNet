import requests
import sys
import os
import time
import io
# import urllib 
# import urllib2 
import json
import time

url_scan = 'https://www.virustotal.com/vtapi/v2/file/scan'
url_report = "https://www.virustotal.com/vtapi/v2/file/report"
apikey = "xxx"

input_dir = "/home/blockchain/Desktop/virustotal/input_dir"
output_dir = "/home/blockchain/Desktop/virustotal/output_dir"
upload_files = {}
path = "/mnt/traffic/xzy/andmal/MalAnd2017/apk"
outputpath = "/mnt/traffic/xzy/andmal/MalAnd2017/upload_virustotal_files"

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def getFileScanId(apikey,a,b):
    params = {'apikey': apikey}
    files = {'file': (a, open(b, 'rb'))}
    response = requests.post(url_scan, files=files, params=params)
    if response.json()['response_code'] == 1:
        my_scan_id = str(response.json()['scan_id'])
        upload_files[a] = my_scan_id
    else:
        print(response.json())
    return my_scan_id

def getFieReportResult(apikey,my_scan_id,filename):
    get_params = {'apikey': apikey, 'resource': my_scan_id,'allinfo': '1'}
    response2 = requests.get(url_report, params=get_params)
    jsondata = json.loads(response2.text)
    data_json = json.dumps(jsondata, cls=NpEncoder)
    fileObject = open(os.path.join(output_dir, os.path.splitext(filename)[0] + '.json'), 'w')
    fileObject.write(data_json)
    fileObject.close()
    
    return jsondata

def getResult(json):
    result = {}
    if json["response_code"] == 1:
        for k,v in json["scans"].items():
            result[k] = v['detected']
        print(result)
    else:
        print(json)

def apks_getResults():
    apk_files_path = None

    upload_files_logs_dir = os.path.join(outputpath, 'upload_files_logs')
    success_upload_files = os.listdir(upload_files_logs_dir)

    detection_results_dir = os.path.join(outputpath, 'detection_results')
    detection_results_files = os.listdir(detection_results_dir)

    params = {'apikey': apikey}

    for i in range(len(success_upload_files)):
        request_file = success_upload_files[i]
        print(str(i) + " " + request_file)
        if request_file in detection_results_files:
            continue

        success_upload_file_path = os.path.join(upload_files_logs_dir, request_file)
        
        success_upload_file = None
        with open(success_upload_file_path) as fp:
            success_upload_file = json.load(fp)
        
        try:
            if 'scan_id' in success_upload_file:
                scan_id = success_upload_file['scan_id']

                get_params = {'apikey': apikey,'resource': scan_id,'allinfo': '1'}
                response = requests.get(url_report, params=get_params)
                jsondata = json.loads(response.text)
                if 'response_code' in jsondata and jsondata['response_code'] == 1:
                    data_json = json.dumps(jsondata, cls=NpEncoder)
                    fileObject = open(os.path.join(detection_results_dir, request_file), 'w')
                    fileObject.write(data_json)
                    fileObject.close()
        except:
            print(response)
        
        time.sleep(18)

def apks_upload():
    apk_files_path = None
    
    with open(os.path.join(outputpath, 'apk_files_path.json')) as fp:
        apk_files_path = json.load(fp)
    
    success_upload_files = []

    upload_files_logs_dir = os.path.join(outputpath, 'upload_files_logs')
    success_upload_files = os.listdir(upload_files_logs_dir)
    success_upload_files = [os.path.splitext(aa)[0] + '.apk' for aa in success_upload_files]


    too_large_apks_dir = os.path.join(outputpath, 'too_large_apks')
    too_large_apks = os.listdir(too_large_apks_dir)
    too_large_apks = [os.path.splitext(aa)[0] + '.apk' for aa in too_large_apks]

    params = {'apikey': apikey}

    for i in range(len(apk_files_path)):
        apk_name = apk_files_path[i][0]
        print(str(i) + ": " + apk_name)
        apk_path = apk_files_path[i][1]

        if apk_name in success_upload_files or apk_name in too_large_apks:
            continue
        try:
            files = {'file': (apk_name, open(apk_path, 'rb'))}
            response = requests.post(url_scan, files=files, params=params)
            res_json = response.json()
            if 'response_code' in res_json and res_json['response_code'] == 1:
                data_json = json.dumps(res_json, cls=NpEncoder)
                fileObject = open(os.path.join(upload_files_logs_dir, os.path.splitext(apk_name)[0] + '.json'), 'w')
                fileObject.write(data_json)
                fileObject.close()
                # my_scan_id = str(response.json()['scan_id'])
                # upload_files[a] = my_scan_id
            else:
                print(response.json())
        except:
            print(response)
            if response.status_code == 413:
                data_json = json.dumps({}, cls=NpEncoder)
                fileObject = open(os.path.join(too_large_apks_dir, os.path.splitext(apk_name)[0] + '.json'), 'w')
                fileObject.write(data_json)
                fileObject.close()


def get_apks_paths():
    print("strart get_apks_paths...")
    apk_files_path = []
    class_name_set = ['Adware', 'Ransomware','Scareware','SMSmalware','Benign'] #,
    for class_name in class_name_set:
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            for family_name in os.listdir(class_path):
                family_path = os.path.join(class_path, family_name)
                if os.path.isdir(family_path):
                    for sample_name in os.listdir(family_path):
                        sample_path = os.path.join(family_path, sample_name)
                        if os.path.splitext(sample_path)[1]=='.apk':
                            apk_files_path.append((sample_name, sample_path, class_name, family_name))
    data_json = json.dumps(apk_files_path, cls=NpEncoder)
    fileObject = open(os.path.join(outputpath, 'apk_files_path.json'), 'w')
    fileObject.write(data_json)
    fileObject.close()

def main():
    for apk_name in os.listdir(input_dir):
        apk_path = os.path.join(input_dir, apk_name)
        if os.path.splitext(apk_path)[1]=='.apk':  
            scan_id = getFileScanId(apikey,apk_name,apk_path)
            time.sleep(2)
            json = getFieReportResult(apikey,scan_id,apk_name)
            getResult(json)

def get_datailed_result():

    detection_results_dir = os.path.join(outputpath, 'detection_results')

    virustotal_detection_results = {}

    interdetector = set()
    uniondetector = set()

    detection_results_files = os.listdir(detection_results_dir)
    
    for i in range(len(detection_results_files)):
        file = detection_results_files[i]
        file_path = os.path.join(detection_results_dir, file)

        file_result = None
        with open(file_path) as fp:
            file_result = json.load(fp)

        result = {}
        for k,v in file_result["scans"].items():
            result[k] = v['detected']
        
        virustotal_detection_results[os.path.splitext(file)[0]] = result

        detector_set = set(list(result.keys()))

        if i == 0:
            interdetector = detector_set
        else:
            interdetector = interdetector & detector_set
        
        uniondetector = uniondetector | detector_set

    data_json = json.dumps(virustotal_detection_results, cls=NpEncoder)
    fileObject = open(os.path.join(outputpath, 'virustotal_detection_results.json'), 'w')
    fileObject.write(data_json)
    fileObject.close()

    detectors_info = {}
    detectors_info['interdetector'] = list(interdetector)
    detectors_info['uniondetector'] = list(uniondetector)

    data_json = json.dumps(detectors_info, cls=NpEncoder)
    fileObject = open(os.path.join(outputpath, 'detectors_info.json'), 'w')
    fileObject.write(data_json)
    fileObject.close()

def filter_result():

    file_result = None
    with open(os.path.join(outputpath, 'virustotal_detection_results.json')) as fp:
        file_result = json.load(fp)

    detectors_info = None
    with open(os.path.join(outputpath, 'detectors_info.json')) as fp:
        detectors_info = json.load(fp)
    
    uniondetector = set(detectors_info['uniondetector'])

    for k,v in file_result.items():
        detector = set(list(v.keys()))
        for detect,detect_val in v.items():
            if detect_val:
                file_result[k][detect] = 1
            else:
                file_result[k][detect] = 0
        for dete in (uniondetector - detector):
            file_result[k][dete] = 2
        if len(file_result[k].keys()) != 70:
            print(k)

    data_json = json.dumps(file_result, cls=NpEncoder)
    fileObject = open(os.path.join(outputpath, 'virustotal_detection_results_union_70.json'), 'w')
    fileObject.write(data_json)
    fileObject.close()


from sklearn import metrics

def get_acc_result():

    apk_files_path = None
    with open(os.path.join(outputpath, 'apk_files_path.json')) as fp:
        apk_files_path = json.load(fp)

    true_lables_map = {}
    for i in range(len(apk_files_path)):
        if apk_files_path[i][2] == 'Benign':
            true_lables_map[os.path.splitext(apk_files_path[i][0])[0]] = 0
        else:
            true_lables_map[os.path.splitext(apk_files_path[i][0])[0]] = 1
        
    file_result = None
    with open(os.path.join(outputpath, 'virustotal_detection_results_union_70.json')) as fp:
        file_result = json.load(fp)

    true_lable = []
    detectors_result = {}

    dd = 0

    for k,v in file_result.items():
        if dd == 0:
            for detector in v.keys():
                detectors_result[detector] = []
            dd += 1
        true_lable.append(true_lables_map[k])
        for detector,detector_val in v.items():
            detectors_result[detector].append(detector_val)

    # ss = 1
    detect_result = []
    for k,v in detectors_result.items():
        for i in range(len(v)):
            if v[i] == 2:
                if true_lable[i] == 0:
                    v[i] = 1
                else:
                    v[i] = 0
        oa = metrics.accuracy_score(true_lable, v)
        recall = metrics.recall_score(true_lable, v, average="binary")
        precision = metrics.precision_score(true_lable, v, average="binary")
        f1 = metrics.f1_score(true_lable, v, average="binary")
        detect_result.append((k, precision, recall, f1, oa))

    detect_result.sort(key=lambda x: (x[4], x[3]), reverse=True)

    data_json = json.dumps(detect_result, cls=NpEncoder)
    fileObject = open(os.path.join(outputpath, 'pre_rec_f1_oa_union_70.json'), 'w')
    fileObject.write(data_json)
    fileObject.close()

if __name__ == '__main__':
    # main()
    # get_apks_paths()
    # apks_upload()
    # apks_getResults()
    # get_datailed_result()
    # filter_result()
    # match_result()
    get_acc_result()
