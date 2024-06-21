SELECT
    DISTINCT YEAR(IsolateDate)
FROM
    tblIsolates
WHERE
    IsolateID IN (
        SELECT IsolateID
        FROM tblRefLink
        WHERE RefID IN (
            SELECT RefID
            FROM tblReferences
            WHERE MedlineID = {pubmed_id}
        )
    )
ORDER BY IsolateDate
;