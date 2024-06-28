SELECT
    CASE
        WHEN COUNT(1) > 0 THEN 'Yes'
        ELSE 'No'
    END
FROM
    tblRxHistory rx,
    tblIsolates iso
WHERE
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
    AND
    rx.RegimenName != 'None'
;
