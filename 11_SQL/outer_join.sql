select 
	a.id as id_a,
	a.fruit as fruit_a,
	a.id as id_b
from 
	basket_a a
left join basket_b b 
on a.fruit = b.fruit;


select 
	a.id as id_a,
	a.fruit as fruit_a,
	b.id as id_b,
	b.fruit as fruit_b
from
	basket_a a
left join basket_b b 
on a.fruit = b.fruit 
where 
	b.id is null;


select 
	a.id as id_a,
	a.fruit as fruit_a,
	b.id as id_b,
	b.fruit as fruit_b
from
	basket_a a
right join basket_b b
on a.fruit = b.fruit;


select 
	a.id as id_a,
	a.fruit as fruit_a,
	b.id as id_b,
	b.fruit as fruit_b
from 
	basket_a a
right join basket_b b 
on a.fruit = b.fruit
where 
	a.id is null;


desc;

select * from information_schema.tables;
select * from information_schema.columns;


select fruit as f from basket_a;

select fruit f from basket_a;



select * from product;



select
	*
from 
	basket_a
order by fruit desc
where row_number() <=2;

cast(product_date as date)



select 
	row_number() over(order by fruit),
	fruit
from 
	basket_a;

select * from basket_a;