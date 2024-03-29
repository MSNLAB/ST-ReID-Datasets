from extractor import ethz, prid2011, market1501, pku, msmt17, duke, cuhk03, personx

dataset_name = {
    "ethz": ethz.Extractor,
    "prid2011": prid2011.Extractor,
    "market1501": market1501.Extractor,
    "pku": pku.Extractor,
    "msmt17": msmt17.Extractor,
    "duke": duke.Extractor,
    "cuhk03": cuhk03.Extractor,
    "personx": personx.Extractor,
}
