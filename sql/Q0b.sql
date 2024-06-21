SELECT
    COUNT(1)
FROM
    tblInVitroSelection
WHERE
    RefID IN (
        SELECT RefID
        FROM tblReferences
        WHERE MedlineID = {pubmed_id}
    )
;
