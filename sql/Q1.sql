SELECT
    CASE
        WHEN COUNT(1) > 0 THEN 'Yes'
        ELSE 'No'
    END
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
            AND
            RefID != 169
        )
    )
;
