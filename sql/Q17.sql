SELECT
    DISTINCT ludrug.DrugClass
FROM
    tblLUDrugs ludrug
WHERE
    DrugName IN (
        SELECT
            DISTINCT drug.DrugName
        FROM
            tblDrugRegimens drug,
            tblRxHistory rx,
            tblIsolates iso
        WHERE
            drug.RxHistoryID = rx.RxHistoryID
            AND
            rx.PtID = iso.PtID
            AND
            iso.IsolateID IN (
                SELECT IsolateID
                FROM tblRefLink
                WHERE RefID IN (
                    SELECT RefID
                    FROM tblReferences
                    WHERE MedlineID = {pubmed_id}
                    AND
                    RefID != 169
                )
            )
            AND
            iso.IsolateDate >= rx.StartDate
    )
;
