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
        )
    )
    AND
    iso.IsolateDate >= rx.StartDate

;
