CREATE TABLE TB_ITEM_COALESCE_TEST
(
ID SERIAL PRIMARY KEY
, PRODUCT VARCHAR (100) NOT NULL
, PRICE NUMERIC NOT NULL
, DISCOUNT NUMERIC
);
INSERT INTO TB_ITEM_COALESCE_TEST
(PRODUCT, PRICE, DISCOUNT)
VALUES
('A', 1000 ,10),
('B', 1500 ,20),
('C', 800 ,5),
('D', 500, NULL);
COMMIT;


select 
	product,
	(price - discount) as net_price
from
	tb_item_coalesce_test;
	
select * from tb_item_coalesce_test;


select
	product,
	(price - coalesce(discount, 0)) as net_price
from
	tb_item_coalesce_test;
	

select 
	product,
	( price -
				case 
				when discount is null then 0
				else discount
				end) as net_price
from
	tb_item_coalesce_test;